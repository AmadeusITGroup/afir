import io
import json
import logging
import os.path
from types import SimpleNamespace

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

from src.brief_prompt import DEFAULT_BRIEF_CHAR_BUDGET
from src.brief_prompt import phrase as _phrase
from src.brief_prompt import render_brief_for_prompt
from src.human_guidance import guidance_message
from src.identity import owner_scoped
from src.link_escalation import ESCALATING_MODES
from src.models.pydantic_models import STUB_OBSERVED, InvestigationReport
from src.utils.llm_client import is_truncation_error
from src.utils.paths import exports_dir

logger = logging.getLogger(__name__)

# Defaults for sizing the single report LLM call. These bound what the LLM narrates;
# every anomaly still lands in the exported JSON/CSV and the PDF/txt anomalies table.
_DEFAULT_MAX_ANOMALIES_IN_PROMPT = 15
_DEFAULT_LLM_INPUT_CHAR_BUDGET = 15000  # mirrors anomaly_detection / correlation


def _containment_withheld(brief) -> bool:
    """True when every subject in the verdict has no lock_target. MagicMock-safe."""
    verdict = getattr(brief, "verdict", None) if brief is not None else None
    subjects = getattr(verdict, "subjects", None) if verdict is not None else None
    if not isinstance(subjects, list) or not subjects:
        return False
    for s in subjects:
        target = getattr(s, "lock_target", None)
        if isinstance(target, dict) and any(
            str(v).strip() for k, v in target.items() if k != "source"
        ):
            return False
    return True


class ReportGenerationModule:
    def __init__(self, config, llm_client, rag=None, knowledge_pack=None, storage=None):
        self.config = config
        self.llm_client = llm_client
        # None means local disk under exports_dir().
        self.storage = storage
        # None is fully supported: every call site passes a generic default sentence.
        self.knowledge_pack = knowledge_pack
        self.rag = rag
        # Degradation channel (read by src/stage_health.py); reset on every call.
        self.last_fallback_used = False
        self.last_backfilled_sections: list = []
        self.last_error = ""
        # A retried truncation is not a failure but signals report_max_tokens is too small.
        self.last_truncation_retried = False
        self.system_prompt = self._build_system_prompt()
        self.user_template = (
            "Incident:\n{incident}\n\nUnderstanding:\n{understanding}\n\n"
            "Investigation Evidence:\n{correlation}\n\nAnomalies:\n{anomalies}"
        )

    @classmethod
    def _build_system_prompt(cls) -> str:
        """Narration instruction. Built from ``_REQUIRED_SECTIONS`` so titles cannot drift."""
        return (
            "You are a fraud investigation analyst writing the final report for a HUMAN "
            "reviewer.\n\n"
            "YOUR JOB IS SYNTHESIS, NOT INVESTIGATION. The investigation is already "
            "finished: the stages before you retrieved the data, correlated it, evaluated "
            "each condition of the applicable procedure, reached a verdict, and scored the "
            "anomalies. Their conclusions are given to you as inputs. You RE-EXPRESS those "
            "conclusions as connected, readable prose that a reviewer can follow — you "
            "explain, order, and join them. You do NOT re-decide them.\n\n"
            "Three prohibitions follow from that, and they are absolute:\n"
            "1. Do NOT INVENT. Every fact, actor, identifier, amount, timestamp and count "
            "in your text must appear in the inputs. If something a reader would want is "
            "not in the inputs, write that it is not available — never supply it.\n"
            "2. Do NOT REVERSE. Where an input states which way a fact points, that "
            "direction is fixed. A check labelled as arguing AGAINST fraud may not be "
            "re-described as suspicious, as a fraud indicator, or as a 'pattern', however "
            "plausible that reading is on its own. If a finding surprises you, report it as "
            "given and say plainly that it is unexpected.\n"
            "3. Do NOT RE-RANK. The decisive condition(s), the verdict, the severity and "
            "each anomaly's confidence are given. Use them as given; never promote, demote, "
            "or aggregate them into a judgement of your own.\n\n"
            "Write for a reviewer who must be able to check you: prefer a specific "
            "identifier over a category, name the source a fact came from, and keep "
            "sentences short enough to be verified one at a time.\n\n"
            "SECTIONS. Return exactly these sections, each a SEPARATE object in the "
            "sections list, in this order, with these exact section_title values:\n"
            f"1. '{cls._REQUIRED_SECTIONS[0]}' — 4-6 sentences, the only part many readers "
            "will read: what the alert claimed, what the investigation concluded (the "
            "verdict, verbatim in meaning), the single most important reason for that "
            "conclusion, and what happens next. No caveats, no method, no lists.\n"
            f"2. '{cls._REQUIRED_SECTIONS[1]}' — what was examined and how far the evidence "
            "reaches: which entities were treated as subjects, the time window, and which "
            "data sources answered. State the LIMITS here (a source that returned nothing, "
            "a truncated result set, a condition that could not be evaluated) because every "
            "later section is read against them. Do not list per-source row counts — a "
            "deterministic section below tabulates them.\n"
            f"3. '{cls._REQUIRED_SECTIONS[2]}' — THE CHRONOLOGY. Narrate what happened in "
            "TIME ORDER, one step at a time, starting from the earliest event in the "
            "evidence and ending with the alert firing. For each step: when, WHO acted "
            "(the identity as the data records it), what they did, and IN WHICH SOURCE it "
            "is recorded — name the source, so the step can be traced to the evidence "
            "files. Where two sources record the same identifier, say so explicitly and "
            "name both: those cross-source joins are what make this one episode rather than "
            "unrelated records, and they are the reason a reviewer trusts the sequence. "
            "Where a step's timing is the point (a value-bearing action minutes after the "
            "record was created, an access before any authentication), state the interval. "
            "This section is "
            "narrative, not a verdict: no conclusion about fraud belongs in it.\n"
            f"4. '{cls._REQUIRED_SECTIONS[3]}' — the analysis: WHY the evidence supports the "
            "conclusion. Take each decisive condition and each scored anomaly and explain, "
            "in plain language, what was checked, what the data showed, and what that means "
            "— including the ones that argue against the alert. Do NOT reproduce the "
            "condition table; it is printed deterministically below. Explain the handful of "
            "results that carry the outcome, and name each anomaly's confidence as given.\n"
            f"5. '{cls._REQUIRED_SECTIONS[4]}' — what is at stake, stated in what the "
            "evidence measured: which documents/entities are affected, their state, the "
            "monetary exposure WITH ITS CURRENCY where the data gives one, and whether the "
            "impact extends beyond what the alert named. If the verdict clears the subject, "
            "the honest content of this section is that there is no exposure — write that in "
            "one sentence rather than speculating about what would be at stake if it had "
            "not.\n"
            f"6. '{cls._REQUIRED_SECTIONS[5]}' — what the reviewer should DO, derived from "
            "the findings and targeting the actual identities and entities in the evidence. "
            "Never recommend evidence-gathering this system already performed (retrieving "
            "logs, correlating sources, pulling the records/sessions) — that data is in this "
            "report. No boilerplate: if the evidence supports no concrete action, say so.\n\n"
            "EACH FACT BELONGS IN EXACTLY ONE SECTION. Overlap between sections is a "
            "defect, not thoroughness: two statements of one fact mean a reader must "
            "reconcile them, and the weaker one is an unchecked paraphrase of the stronger. "
            "So: the sequence of events lives ONLY in the reconstruction; the reasoning "
            "lives ONLY in the analysis; the consequences live ONLY in impact; the actions "
            "live ONLY in next steps. The summary is allowed to state the conclusion the "
            "other sections support — that is its purpose — but nothing else may be "
            "repeated. Refer back ('as the chronology shows') rather than restating.\n\n"
            "SECTIONS YOU MUST NOT WRITE. These are appended automatically, built "
            "deterministically from checked values — the alert's verbatim facts, the "
            "reconciliation of those facts against the data, the per-condition assessment "
            "table, the verdict, the wider-evidence sweep, the evidence coverage list, the "
            "authorised actions, and the evidence-file appendix. A narrated second copy of "
            "any of them would be an unverified paraphrase of a checked fact. Read them, "
            "rely on them, cite them in your prose — do not reproduce them as sections, and "
            "do not use their titles.\n\n"
            "VERDICT AUTHORITY: when an AUTHORITATIVE INVESTIGATION BRIEF is provided, its "
            "verdict is ground truth. State the outcome and the DECISIVE condition(s) the "
            "brief names — never a reason of your own choosing. Recommend containment ONLY "
            "where the brief's action backbone does: if it nominates no containment target, "
            "then no lock, suspension, void or external notification may appear anywhere in "
            "your text, in any form, including as a conditional or a precaution. If the "
            "verdict clears the subject, say so plainly and stop. If it is INSUFFICIENT "
            "DATA, say exactly what is missing and that a decision is pending. Where "
            "past-investigation precedents are supplied, note how comparable cases were "
            "resolved.\n\n"
            "REPORTING DISCIPLINE (each of these has caused a defective report):\n"
            "- Label an INFERENCE as an inference, in the sentence that makes it, and keep "
            "it out of the findings. A finding is something the evidence shows.\n"
            "- No claim resting on an unverified inference may carry a confidence above LOW.\n"
            "- Never quote a count from a result set marked truncated/row-limited: state a "
            "lower bound and label it as one.\n"
            "- A ROW count is never a count of business objects and never a monetary value.\n"
            "- Every amount carries its currency, or is not stated as an amount.\n"
            "- Never attribute an artefact to a party whose identifier range does not match "
            "it.\n"
            "- Use the incident's own severity as given. Do NOT invent a numeric severity "
            "scale or score.\n"
            "- A condition that could not be evaluated is NOT a condition that passed; say "
            "which ones they are. And the ban is on the PROPERTY, not just on the word "
            "'pass': whatever a NOT EVALUATED condition was there to establish may not be "
            "asserted anywhere in your text, in any wording, however strongly the rest of "
            "the evidence suggests it. If the check that would have identified WHAT KIND OF "
            "PARTY acted was not evaluated, the actor's kind is unknown — do not call it "
            "one; if a check on an entity's state was not evaluated, that state is unknown. "
            "Where such a property matters to the reader, name the condition and say it "
            "could not be answered. (A live report called the subject 'the identified "
            "automation' in its summary while the one check that identifies automation had "
            "returned no rows.)\n\n"
            "Return the report as a list of sections; each has a section_title and content, "
            "where content is a string or a list of strings. All six must be present."
        )

    def _report_phrases(self, use_case=None) -> dict:
        """The pack's ``reporting.phrases`` map for this use case; ``{}`` if unavailable."""
        pack = getattr(self, "knowledge_pack", None)
        if pack is None:
            return {}
        try:
            phrases = pack._reporting_for(use_case, "phrases")
        except Exception:  # a pack stub / mock without the accessor
            return {}
        return phrases if isinstance(phrases, dict) else {}

    def _phase_rules(self, use_case=None):
        """The pack's chronology phase rules, else ``_GENERIC_PHASE_RULES``."""
        pack = getattr(self, "knowledge_pack", None)
        if pack is not None:
            try:
                rules = pack.phase_rules(use_case)
            except Exception:
                rules = None
            if rules:
                return [(tuple(r["keywords"]), r["label"]) for r in rules]
        return self._GENERIC_PHASE_RULES

    # Hard cap for the one truncation retry, so a pathological prompt cannot escalate
    # without limit.
    _TRUNCATION_RETRY_CEILING = 16000

    def _report_max_tokens(self, prompt_anomalies):
        """Response budget for the narration call: configured ceiling, floored by ``report_min_tokens``."""
        floor = self.config.get("report_min_tokens", 1500)
        ceiling = self.config.get("report_max_tokens", self.llm_client.max_tokens)
        return max(floor, ceiling)

    @staticmethod
    def render_markdown(sections, incident, anomalies=None):
        """Render section list as Markdown. Pure formatting; never raises on odd shapes."""
        lines = [
            f"# Fraud Investigation Report — Incident {incident.get('id', '')}".rstrip(),
            "",
        ]

        def emit(content, depth=0):
            if content is None:
                return
            if isinstance(content, (str, int, float)):
                text = str(content).strip()
                if text:
                    lines.append(text)
                    lines.append("")
            elif isinstance(content, dict):
                title = content.get("section_title")
                if title:
                    lines.append(f"{'#' * min(depth + 2, 6)} {title}")
                    lines.append("")
                if "content" in content:
                    emit(content["content"], depth + 1)
            elif isinstance(content, list):
                # A list of scalars renders as bullets; a list of dicts recurses.
                if all(isinstance(x, (str, int, float)) for x in content):
                    for item in content:
                        text = str(item).strip()
                        if text:
                            # Avoid a double bullet when the item already starts with a
                            # dash (some sections pre-format their own list markers).
                            text = text[1:].strip() if text.startswith("- ") else text
                            lines.append(f"- {text}")
                    lines.append("")
                else:
                    for item in content:
                        emit(item, depth)

        for section in sections or []:
            title = (
                (section or {}).get("section_title", "")
                if isinstance(section, dict)
                else ""
            )
            if title:
                lines.append(f"## {title}")
                lines.append("")
            # depth=1 so a nested subsection dict renders at ### (below the ## section).
            emit(
                (
                    (section or {}).get("content")
                    if isinstance(section, dict)
                    else section
                ),
                1,
            )

        return "\n".join(lines).rstrip() + "\n"

    def _write_readable_reports(self, sections, incident, anomalies):
        """Best-effort write Markdown + PDF next to the JSON export. Never raises."""
        incident_id = incident.get("id", "report")
        # Scoped to the run's owner, so the readable report lands beside the JSON export
        # rather than in a shared namespace the exports left.
        backend = owner_scoped(self._artifact_storage(), incident)
        try:
            md = self.render_markdown(sections, incident, anomalies)
            backend.put_text(f"fraud_report_{incident_id}.md", md)
        except Exception as e:
            logger.error("Markdown report render failed for %s: %s", incident_id, e)
        try:
            pdf = self.generate_pdf_report(sections, incident, anomalies or [])
            backend.put_bytes(f"fraud_report_{incident_id}.pdf", pdf)
        except Exception as e:
            logger.error("PDF report render failed for %s: %s", incident_id, e)

    def _artifact_storage(self):
        """Storage backend for the readable reports; local disk under ``exports_dir()`` by default."""
        if getattr(self, "storage", None) is not None:
            return self.storage
        from src.storage import LocalStorage

        return LocalStorage(root=exports_dir())

    def append_pdf_content(self, story, content, styles):
        if isinstance(content, (str, int, float)):
            story.append(Paragraph(str(content), styles["Normal"]))
        elif isinstance(content, dict):
            title = content.get("section_title")
            if title:
                story.append(Paragraph(title, styles["Heading2"]))
            if "content" in content:
                self.append_pdf_content(story, content["content"], styles)
        elif isinstance(content, list):
            for item in content:
                self.append_pdf_content(story, item, styles)
        return story

    async def generate(
        self, incident, understanding, logs, anomalies, correlation=None, guidance=None
    ):
        """Produce the investigation report; falls back to a deterministic report on LLM failure."""
        max_in_prompt = self.config.get(
            "max_anomalies_in_prompt", _DEFAULT_MAX_ANOMALIES_IN_PROMPT
        )
        char_budget = self.config.get(
            "llm_input_char_budget", _DEFAULT_LLM_INPUT_CHAR_BUDGET
        )
        # Sub-threshold items are marked, not dropped; `ranked` drives the narrative only.
        narratable = [
            a for a in anomalies if not bool(getattr(a, "below_threshold", False))
        ]
        ranked = sorted(narratable, key=lambda a: a.confidence_score, reverse=True)
        prompt_anomalies = ranked[:max_in_prompt]
        omitted = len(ranked) - len(prompt_anomalies)

        # Reset degradation channel for this call.
        self.last_fallback_used = False
        self.last_backfilled_sections = []
        self.last_error = ""
        self.last_truncation_retried = False

        sections = await self._narrate_sections(
            incident,
            understanding,
            correlation,
            prompt_anomalies,
            omitted,
            char_budget,
            guidance,
        )
        if sections is None:
            self.last_fallback_used = True
            sections = self._fallback_sections(
                incident, understanding, logs, ranked, correlation
            )
        else:
            sections = self._ensure_complete_sections(
                sections, incident, understanding, logs, ranked, correlation
            )
            # Skipped when "Incident Reconstruction" was backfilled (backfill is this content).
            self._attach_timeline(sections, correlation, incident.get("id"))

        phrases = self._report_phrases(self._use_case_of(correlation))
        for builder in (
            self._alert_facts_section,
            self._reconciliation_section,
            self._conditions_section,
            self._wider_evidence_section,
            self._verdict_section,
            self._not_retrieved_section,
            self._actions_section,
            # Built here so it runs on both the narrated and fallback paths; adding it
            # to `_fallback_sections` as well would emit it twice on the fallback path.
            self._links_section,
            self._inquiries_section,
        ):
            try:
                section = builder(correlation, phrases)
            except Exception as e:
                logger.warning(
                    "Deterministic report section %s failed for incident %s: %s",
                    getattr(builder, "__name__", builder),
                    incident.get("id"),
                    e,
                )
                section = None
            if section is not None:
                sections.append(section)

        # Sub-threshold findings must be noted: a clean incident and one just under the
        # threshold read identically without this. Merged into the coverage section.
        self._note_subthreshold(sections, anomalies)

        # Filenames are deterministic from incident id; export runs later.
        sections.append(self._evidence_artifacts_section(incident.get("id")))
        sections = self._order_sections(sections)

        # Best-effort: never fails the stage.
        self._write_readable_reports(sections, incident, anomalies)

        try:
            if self.config["output_format"] == "pdf":
                pdf = self.generate_pdf_report(sections, incident, anomalies)
                self.export_report(pdf, "pdf", self.config["output_path"])
                return pdf
            else:
                text = json.dumps(sections, indent=2)
                self.export_report(text, "txt", self.config["output_path"])
                return text
        except Exception as e:
            # Return JSON sections so the caller has a report object, not a hard failure.
            logger.error(
                "Report rendering failed for incident %s (%s); returning JSON sections.",
                incident.get("id"),
                e,
            )
            return json.dumps(sections, indent=2)

    # Kept as a method so tests can call it by name here.
    _render_brief_for_prompt = staticmethod(render_brief_for_prompt)

    def _brief_char_budget(self):
        """Chars the brief may take from this prompt; same key as anomaly detection."""
        return int(
            self.config.get("brief_char_budget", DEFAULT_BRIEF_CHAR_BUDGET)
        )

    async def _narrate_sections(
        self,
        incident,
        understanding,
        correlation,
        prompt_anomalies,
        omitted,
        char_budget,
        guidance=None,
    ):
        """Ask the LLM to narrate the report sections; return None on any failure."""
        try:
            evidence = getattr(correlation, "evidence", None) if correlation else None
            if evidence is not None:
                from src.evidence import render_for_prompt

                correlation_text = render_for_prompt(evidence, char_budget)
                if correlation is not None and correlation.summary_text:
                    correlation_text = (
                        f"Correlation summary: {correlation.summary_text}\n\n"
                        + correlation_text
                    )
            else:
                correlation_text = (
                    correlation.model_dump_json(indent=2)
                    if correlation is not None
                    else "No correlation summary available."
                )
                if len(correlation_text) > char_budget:
                    correlation_text = (
                        correlation_text[:char_budget] + "... [truncated]"
                    )

            anomalies_text = json.dumps(
                [a.model_dump() for a in prompt_anomalies], indent=2
            )
            if omitted > 0:
                anomalies_text += (
                    f"\n\n[{omitted} additional lower-confidence anomalies omitted from "
                    "this prompt for brevity; all appear in the attached anomalies table.]"
                )

            phrases = self._report_phrases(self._use_case_of(correlation))
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            domain_hint = "\n".join(
                t
                for t in (
                    _phrase(phrases, "narration_concepts", ""),
                    _phrase(
                        phrases,
                        "reconstruction_shape",
                        "",
                        subject_label=self._subject_label(correlation),
                    ),
                )
                if t
            )
            if domain_hint:
                messages.append(
                    {
                        "role": "system",
                        "content": "DOMAIN GUIDANCE for this use case (how to phrase and "
                        "order the narrative — it does not change any finding):\n"
                        + domain_hint,
                    }
                )
            brief = getattr(correlation, "brief", None) if correlation else None
            brief_text = self._render_brief_for_prompt(
                brief, char_budget=self._brief_char_budget(), phrases=phrases
            )
            if brief_text:
                messages.append(
                    {
                        "role": "system",
                        "content": "AUTHORITATIVE INVESTIGATION BRIEF (ground truth — the "
                        "narrative may NOT contradict it):\n" + brief_text,
                    }
                )
            human_msg = guidance_message(guidance)
            if human_msg is not None:
                messages.append(human_msg)
            messages.append(
                {
                    "role": "user",
                    "content": self.user_template.format(
                        incident=json.dumps(incident, indent=2),
                        understanding=understanding.analysis.model_dump_json(indent=2),
                        correlation=correlation_text,
                        anomalies=anomalies_text,
                    ),
                }
            )

            tokens = self._report_max_tokens(prompt_anomalies)
            try:
                report = await self.llm_client.structured_output(
                    messages,
                    response_model=InvestigationReport,
                    max_tokens=tokens,
                    rag=getattr(self, "rag", None),
                    stage="report_generation",
                )
            except Exception as e:
                # Matched on the marker function, not `except LLMTruncatedError`: the
                # module path differs between this module and main.py's import, so the
                # class object raised is not the class object imported here.
                if not is_truncation_error(e):
                    raise
                # Truncation → retry once at wider budget; a second overrun means the prompt.
                retry_tokens = min(tokens * 2, self._TRUNCATION_RETRY_CEILING)
                if retry_tokens <= tokens:
                    raise
                logger.warning(
                    "Report narration truncated at %d tokens (%s); retrying once at %d.",
                    tokens,
                    e,
                    retry_tokens,
                )
                self.last_truncation_retried = True
                report = await self.llm_client.structured_output(
                    messages,
                    response_model=InvestigationReport,
                    max_tokens=retry_tokens,
                    rag=getattr(self, "rag", None),
                    stage="report_generation",
                )
            logger.info("Correctly generated report content")
            return report.model_dump()["sections"]
        except Exception as e:
            self.last_error = str(e)
            logger.error(
                "Report narration LLM call failed for incident %s (%s); "
                "falling back to a deterministic report.",
                incident.get("id"),
                e,
            )
            return None

    # Narrated sections matched by lower-cased keyword; deterministic sections are appended
    # by generate() and ordered by _SECTION_ORDER.
    _REQUIRED_SECTIONS = [
        "Executive Summary",
        "Investigation Scope and Method",
        "Incident Reconstruction",
        "Analysis and Findings",
        "Impact and Exposure",
        "Recommended Next Steps",
    ]

    # --- Deterministic section titles. Three places must agree: builder, ordering table,
    # and the LLM prompt telling the model not to write them.
    _SEC_ALERT_FACTS = "Alert Facts (verbatim from the payload)"
    _SEC_RECONCILIATION = "Reconciliation of Declared Facts Against Retrieved Data"
    _SEC_CONDITIONS = "Condition Assessment (per subject, in procedure order)"
    _SEC_WIDER = "Wider Evidence (scope sweep)"
    _SEC_NOT_RETRIEVED = "Evidence Coverage and Gaps"
    _SEC_ACTIONS = "Authorised Actions"
    # Advisory: questions about other procedures addressed to a human, not part of the verdict.
    _SEC_LINKS = "Cross-Procedure Correlation (advisory — not part of the verdict)"
    # Advisory, and the other axis: what THIS procedure could not settle about its own evidence.
    _SEC_INQUIRIES = "Open Questions This Procedure Left (advisory — not part of the verdict)"
    _SEC_ARTIFACTS = "Evidence Artifacts"
    # Subsection inside Incident Reconstruction, not a top-level section.
    _SUB_TIMELINE = "Source records in time order (built from the correlated evidence)"

    # Canonical order: conclusion first, evidence middle, appendix last.
    # Matched by keyword, longest first, so paraphrased titles land deterministically.
    _SECTION_ORDER = [
        # --- Conclusion -------------------------------------------------------
        "Executive Summary",
        # Pack-driven title ("<scheme> Verdict"), so matched on "verdict".
        "Verdict",
        # --- What was alleged -------------------------------------------------
        _SEC_ALERT_FACTS,
        _SEC_RECONCILIATION,
        # --- How it was investigated ------------------------------------------
        "Investigation Scope and Method",
        _SEC_NOT_RETRIEVED,
        # --- What the evidence shows ------------------------------------------
        "Incident Reconstruction",
        _SEC_CONDITIONS,
        "Analysis and Findings",
        _SEC_WIDER,
        "Impact and Exposure",
        # --- What may be done -------------------------------------------------
        _SEC_ACTIONS,
        _SEC_LINKS,
        _SEC_INQUIRIES,
        "Recommended Next Steps",
        # --- Appendix ---------------------------------------------------------
        _SEC_ARTIFACTS,
    ]

    def _ensure_complete_sections(
        self, sections, incident, understanding, logs, ranked_anomalies, correlation
    ):
        """Backfill any missing required section from the deterministic builder."""
        present = " ".join(
            str((s or {}).get("section_title", "")).lower()
            for s in sections
            if isinstance(s, dict)
        )
        missing = [
            title for title in self._REQUIRED_SECTIONS if title.lower() not in present
        ]
        if missing:
            self.last_backfilled_sections = list(missing)
            logger.info(
                "Report narration omitted %d section(s) %s; backfilling deterministically.",
                len(missing),
                missing,
            )
            det_sections = self._fallback_sections(
                incident, understanding, logs, ranked_anomalies, correlation
            )
            for title in missing:
                key = title.lower()
                sec = next(
                    (
                        s
                        for s in det_sections
                        if key in str(s.get("section_title", "")).lower()
                    ),
                    None,
                ) or self._det_section_for(
                    title, incident, understanding, ranked_anomalies, correlation
                )
                if sec:
                    sections.append(sec)

        return sections

    def _attach_timeline(self, sections, correlation, incident_id):
        """Append the deterministic timeline as a subsection of 'Incident Reconstruction'. Never raises."""
        if "Incident Reconstruction" in (self.last_backfilled_sections or []):
            return
        target = next(
            (
                s
                for s in sections
                if isinstance(s, dict)
                and "incident reconstruction" in str(s.get("section_title", "")).lower()
            ),
            None,
        )
        if target is None:
            return
        try:
            sub = self._timeline_subsection(correlation, incident_id)
        except Exception as e:
            logger.warning("Timeline subsection failed for %s: %s", incident_id, e)
            return
        if sub is None:
            return
        content = target.get("content")
        if isinstance(content, list):
            content.append(sub)
        elif content is None:
            target["content"] = [sub]
        else:
            # Scalar body: wrap so the subsection can sit beside it.
            target["content"] = [content, sub]

    def _det_section_for(
        self, title, incident, understanding, ranked_anomalies, correlation
    ):
        """Deterministic section for a title not covered by ``_fallback_sections``."""
        a = getattr(understanding, "analysis", None)
        evidence = (
            getattr(correlation, "evidence", None) if correlation is not None else None
        )
        if title == "Incident Reconstruction":
            return {
                "section_title": title,
                "content": self._incident_reconstruction(
                    correlation, incident.get("id")
                ),
            }
        if title == "Impact and Exposure":
            return {
                "section_title": title,
                "content": self._fallback_implications(
                    ranked_anomalies, evidence, a, correlation
                ),
            }
        if title == "Recommended Next Steps":
            return {
                "section_title": title,
                "content": self._fallback_recommendations(
                    ranked_anomalies, evidence, a, correlation
                ),
            }
        return None

    def _order_sections(self, sections):
        """Stable-sort sections into ``_SECTION_ORDER``; unrecognised sections sit just before artifacts."""
        order = {t.lower(): i for i, t in enumerate(self._SECTION_ORDER)}
        # Longest keyword first: "recommended next steps" must beat "actions".
        keys = sorted(order, key=len, reverse=True)
        last_idx = order[self._SEC_ARTIFACTS.lower()]

        def rank(item):
            title = str((item[1] or {}).get("section_title", "")).lower()
            for key in keys:
                if key in title:
                    return order[key]
            # Unrecognised sections sit immediately before Evidence Artifacts.
            return last_idx - 0.5

        # Decorate with original index for a stable sort.
        return [
            s for _, s in sorted(enumerate(sections), key=lambda it: (rank(it), it[0]))
        ]

    # Fallback phase rules; vocabulary-free, covers the three phases any access-based incident has.
    _GENERIC_PHASE_RULES = [
        (
            ("auth", "signin", "login", "session", "logon", "oauth", "identification"),
            "Authentication and session activity",
        ),
        (
            ("access", "retrieve", "retriev", "read", "query", "display", "gdpr"),
            "Data access",
        ),
        (("alert", "siem", "detection"), "Alert and detection"),
    ]

    @classmethod
    def _phase_for(cls, source, action, rules=None):
        """Classify a chronology event into a phase from its source/action. First hit wins."""
        hay = f"{source} {action}".lower()
        for keywords, label in rules or cls._GENERIC_PHASE_RULES:
            if any(k in hay for k in keywords):
                return label
        return "Activity"

    def _incident_reconstruction(self, correlation, incident_id, heading=True):
        """Chronological narrative built from the evidence. ``heading=False`` for subsection use."""
        evidence = (
            getattr(correlation, "evidence", None) if correlation is not None else None
        )
        raw_file = f"evidence_raw_{incident_id}.json"
        transformed_file = f"evidence_transformed_{incident_id}.json"
        if evidence is None:
            return [
                "No structured evidence pack was assembled, so a chronological "
                "reconstruction is unavailable. See the raw records in "
                f"{raw_file}."
            ]

        chrono = list(getattr(evidence, "chronology", []) or [])
        subj = [e for e in chrono if getattr(e, "is_subject", False)]
        timeline = subj or chrono
        # Count before the cap; the note after tells the reader how many were omitted.
        dated = [e for e in timeline if getattr(e, "epoch", None) is not None]
        timeline = dated[: self._CHRONO_LINE_CAP]

        out = (
            [
                "The following reconstructs the incident in chronological order from the "
                f"correlated evidence ({transformed_file}); each step cites the source it "
                f"came from so it can be traced to the raw records in {raw_file}.",
            ]
            if heading
            else []
        )
        if not timeline:
            out.append(
                "No time-ordered subject events could be reconstructed; consult the "
                "actor attribution and cross-source joins in the transformed evidence."
            )
            return out

        rules = self._phase_rules(self._use_case_of(correlation))
        last_phase = None
        for ev in timeline:
            phase = self._phase_for(
                getattr(ev, "source", ""), getattr(ev, "action", ""), rules
            )
            if phase != last_phase:
                out.append(f"[{phase}]")
                last_phase = phase
            actor = getattr(ev, "actor", "") or "unknown actor"
            action = getattr(ev, "action", "") or "activity"
            ts = getattr(ev, "timestamp", "") or "unknown time"
            source = getattr(ev, "source", "")
            ents = getattr(ev, "entities", {}) or {}
            ent_bits = ", ".join(
                f"{k}={v}" for k, v in ents.items() if k not in ("time_window",)
            )
            count = getattr(ev, "count", 1) or 1
            times = f" ×{count}" if count > 1 else ""
            detail = f" ({ent_bits})" if ent_bits else ""
            out.append(
                f"- {ts} — {actor} performed '{action}'{times} in {source}{detail}."
            )
        note = self._cut_note(
            len(dated), self._CHRONO_LINE_CAP, "later event(s) in the same chronology"
        )
        if note:
            out.append(note)
        joins = getattr(evidence, "cross_source_joins", []) or []
        if joins:
            tie = "; ".join(
                f"{j.get('value')} seen in {', '.join(j.get('sources', []))}"
                for j in joins[:5]
                if j.get("value")
            )
            if tie:
                out.append(
                    f"Cross-source linkage confirming a single actor's activity: {tie}."
                )
        out.extend(self._asset_impact_lines(correlation))
        return out

    def _timeline_subsection(self, correlation, incident_id):
        """The deterministic chronology as a subsection of 'Incident Reconstruction', or ``None``."""
        lines = self._incident_reconstruction(correlation, incident_id, heading=False)
        # A single "no evidence" line is not a timeline.
        if not lines or len(lines) < 2:
            return None
        return {"section_title": self._SUB_TIMELINE, "content": lines}

    @staticmethod
    def _use_case_of(correlation):
        """The brief's use-case name, or ``None``. MagicMock-safe (must be a real ``str``)."""
        brief = getattr(correlation, "brief", None) if correlation is not None else None
        uc = getattr(brief, "use_case", None) if brief is not None else None
        return uc if isinstance(uc, str) and uc.strip() else None

    @staticmethod
    def _subject_label(correlation):
        """Subject type from the verdict, else ``"subject"``."""
        verdict = (
            getattr(correlation, "verdict", None) if correlation is not None else None
        )
        subjects = getattr(verdict, "subjects", None) if verdict is not None else None
        if isinstance(subjects, list) and subjects:
            stype = getattr(subjects[0], "subject_type", "")
            if isinstance(stype, str) and stype.strip():
                return stype.strip()
        return "subject"

    @classmethod
    def _asset_impact_lines(cls, correlation):
        """Asset-impact lines from the brief's timeline. Returns ``[]`` for a verdict-less incident."""
        brief = getattr(correlation, "brief", None) if correlation is not None else None
        if brief is None:
            return []
        timeline = getattr(brief, "asset_timeline", None)
        timeline = timeline if isinstance(timeline, list) else []
        assets = getattr(brief, "impacted_assets", None)
        assets = assets if isinstance(assets, list) else []
        if not timeline and not assets:
            return []
        lines = ["[Asset impact]"]
        # The scope sweep's per-asset rollup rendered deterministically: `status` is the
        # current state (the highest version of a versioned record), which is what
        # containment turns on.
        if assets:
            scope = getattr(brief, "scope_status", "") or ""
            lines.append(
                "Assets touched by the actor, with their CURRENT state"
                + (f" ({scope})" if scope else "")
                + ":"
            )
            for a in assets[: cls._ASSET_TABLE_CAP]:
                amount = " ".join(
                    str(x)
                    for x in (getattr(a, "amount", ""), getattr(a, "currency", ""))
                    if x
                )
                actor = getattr(a, "actor", "") or ""
                event_date = getattr(a, "event_date", "") or ""
                # Out-of-window assets: adjacent activity, not incident scope; must be labelled.
                if not getattr(a, "in_window", True):
                    flag = "  <-- OUTSIDE the incident window (adjacent activity, not scope)"
                elif not getattr(a, "known", False):
                    flag = "  <-- NOT named in the alert"
                else:
                    flag = ""
                lines.append(
                    f"- {getattr(a, 'asset_id', '') or '(no document)'} on "
                    f"{getattr(a, 'subject', '')}"
                    + (f" — {amount}" if amount else "")
                    + f" — state: {getattr(a, 'status', '') or 'unknown'}"
                    + (f" — issued {event_date}" if event_date else "")
                    + (f" — by {actor}" if actor else "")
                    + flag
                )
            note = cls._cut_note(
                len(assets), cls._ASSET_TABLE_CAP, "asset(s) touched by the same actor"
            )
            if note:
                lines.append(note)
        if not timeline:
            return lines
        lines.append(
            "Impact of the actor's actions on the affected assets, in time order:"
        )
        for e in timeline[: cls._CHRONO_LINE_CAP]:
            ts = getattr(e, "timestamp", "") or "unknown time"
            etype = getattr(e, "event_type", "") or "event"
            ent_type = getattr(e, "entity_type", "") or ""
            ent_val = getattr(e, "entity_value", "") or ""
            actor = getattr(e, "actor", "") or ""
            detail = getattr(e, "detail", "") or ""
            who = f" by {actor}" if actor else ""
            what = f" {ent_type} {ent_val}".rstrip()
            extra = f" — {detail}" if detail and detail.lower() != etype.lower() else ""
            lines.append(f"- {ts} — {etype}{what}{who}{extra}.")
        note = cls._cut_note(
            len(timeline), cls._CHRONO_LINE_CAP, "later asset event(s) in the same order"
        )
        if note:
            lines.append(note)
        return lines

    @staticmethod
    def _readable_correlation_summary(correlation):
        """Human-readable correlation summary; never a raw JSON dump."""
        lines = []
        aggs = getattr(correlation, "aggregations", {}) or {}
        resolved = aggs.get("resolved_correlation_keys", []) or []
        if resolved:
            key_names = ", ".join(
                str(k.get("entity_hint", "?")) for k in resolved if isinstance(k, dict)
            )
            lines.append(f"Correlation keys evaluated: {key_names}.")
        # Cross-source join outcomes from the transforms (matches + the values that hit).
        transforms = getattr(correlation, "transforms", []) or []
        any_join = False
        for t in transforms:
            if getattr(t, "op", "") != "cross_source_overlap":
                continue
            any_join = True
            rows = getattr(t, "rows", []) or []
            label = getattr(t, "label", "join")
            if rows:
                vals = ", ".join(
                    str(r.get("value"))
                    for r in rows[:5]
                    if isinstance(r, dict) and r.get("value") is not None
                )
                srcs = sorted(
                    {
                        s
                        for r in rows
                        if isinstance(r, dict)
                        for s in (r.get("sources") or [])
                    }
                )
                lines.append(
                    f"{label}: {len(rows)} match(es)"
                    + (f" on {vals}" if vals else "")
                    + (f" across {', '.join(srcs)}" if srcs else "")
                    + "."
                )
            else:
                lines.append(f"{label}: no cross-source matches.")
        if any_join and not any("match" in ln for ln in lines):
            lines.append("No cross-source correlations were found.")
        if not lines:
            # Last resort: a non-JSON summary_text.
            raw = getattr(correlation, "summary_text", "") or ""
            if isinstance(raw, str) and raw and not raw.lstrip().startswith(("{", "[")):
                lines.append(raw)
        return lines or ["No correlation summary available."]

    def _fallback_sections(
        self, incident, understanding, logs, ranked_anomalies, correlation
    ):
        """Deterministic report built from data in hand (no LLM). Mirrors the narrated structure."""
        a = getattr(understanding, "analysis", None)
        summary = getattr(a, "incident_summary", "") if a else ""
        severity = (getattr(a, "severity", "") if a else "") or ""
        entities = getattr(a, "extracted_entities", []) if a else []
        entity_lines = [
            f"{getattr(e, 'type', '?')}: {getattr(e, 'value', '')}"
            for e in (entities or [])
        ]
        source_lines = [
            f"{source}: {len(rows or [])} record(s) retrieved"
            for source, rows in (logs or {}).items()
        ]

        record_count = None
        evidence = (
            getattr(correlation, "evidence", None) if correlation is not None else None
        )
        if correlation is not None:
            rc = getattr(correlation, "record_count", None)
            record_count = rc if isinstance(rc, int) else None

        incident_id = incident.get("id")
        process_content = list(source_lines) or ["No log sources returned data."]
        if correlation is not None:
            process_content = (
                [
                    (
                        f"Correlated {record_count} record(s) across the sources below."
                        if record_count is not None
                        else "Correlation ran over the sources below."
                    )
                ]
                + process_content
                + self._readable_correlation_summary(correlation)
            )

        reconstruction = self._incident_reconstruction(correlation, incident_id)
        findings_content = self._fallback_findings(ranked_anomalies, evidence)
        implications = self._fallback_implications(
            ranked_anomalies, evidence, a, correlation
        )
        next_steps = self._fallback_recommendations(
            ranked_anomalies, evidence, a, correlation
        )

        note = (
            "NOTE: This report was assembled deterministically from the retrieved "
            "evidence because the narration model was unavailable at generation time. "
            "All retrieved data, correlation results and detected anomalies are included."
        )
        # Titles mirror _REQUIRED_SECTIONS for keyword-match backfill.
        scope_content = [
            f"Incident ID: {incident_id}",
            # The name this run is filed under everywhere else. Stated beside the id rather
            # than instead of it: the id is what every artifact of this report is keyed on.
            *(
                [f"Run label: {incident['label']}"]
                if incident.get("label")
                else []
            ),
            (
                f"Severity (as stated by the incident): {severity}"
                if severity
                else "Severity: not stated by the incident"
            ),
        ]
        if entity_lines:
            scope_content.append("Subjects and entities examined:")
            scope_content.append(entity_lines)
        scope_content.extend(process_content)
        return [
            {
                "section_title": self._REQUIRED_SECTIONS[0],  # Executive Summary
                "content": [note, summary or "Incident under investigation."],
            },
            {
                "section_title": self._REQUIRED_SECTIONS[1],  # Investigation Scope…
                "content": scope_content,
            },
            {
                "section_title": self._REQUIRED_SECTIONS[2],  # Incident Reconstruction
                "content": reconstruction,
            },
            {
                "section_title": self._REQUIRED_SECTIONS[3],  # Analysis and Findings
                "content": findings_content,
            },
            {
                "section_title": self._REQUIRED_SECTIONS[4],  # Impact and Exposure
                "content": implications,
            },
            {
                "section_title": self._REQUIRED_SECTIONS[5],  # Recommended Next Steps
                "content": next_steps,
            },
        ]

    @staticmethod
    def _fallback_findings(ranked_anomalies, evidence):
        """Deterministic findings body; falls back to an actor/chronology narrative when no anomalies."""
        if ranked_anomalies:
            blocks = []
            for i, an in enumerate(ranked_anomalies, 1):
                conf = getattr(an, "confidence_score", None)
                conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
                desc = getattr(an, "description", "") or "(no description)"
                reason = getattr(an, "patterns", "") or ""
                support = getattr(an, "supporting_data", "") or ""
                body = [f"Finding: {desc}", f"Confidence: {conf_txt}"]
                if reason:
                    body.append(f"Why it was flagged: {reason}")
                if support:
                    body.append(f"Supporting evidence: {support}")
                blocks.append(
                    {
                        "section_title": f"Anomaly {i} (confidence {conf_txt})",
                        "content": body,
                    }
                )
            return blocks
        if evidence is None:
            return ["No anomalies were scored and no evidence was assembled."]

        out = [
            "No anomalies were scored by the model; the following is a deterministic "
            "reconstruction from the correlated evidence.",
        ]
        actors = getattr(evidence, "actors", []) or []
        subjects = [a for a in actors if getattr(a, "is_subject", False)]
        background = [a for a in actors if not getattr(a, "is_subject", False)]

        def _fmt_actor(act):
            actions = ", ".join(
                f"{k}={v}" for k, v in (getattr(act, "action_counts", {}) or {}).items()
            )
            span = f"{act.first_seen or '?'} .. {act.last_seen or '?'}"
            touched = "; ".join(
                f"{k}={','.join(v)}"
                for k, v in (getattr(act, "entities_touched", {}) or {}).items()
            )
            return (
                f"- {act.actor}: {act.event_count} event(s) [{span}] across "
                f"{', '.join(act.sources)}"
                + (f"; actions: {actions}" if actions else "")
                + (f"; entities: {touched}" if touched else "")
            )

        if subjects:
            out.append("Persons of interest (from the incident):")
            for act in subjects[:5]:
                out.append(_fmt_actor(act))
        if background:
            out.append(
                "Other identities active in the same window (context, not implicated):"
            )
            for act in background[:5]:
                out.append(_fmt_actor(act))
        if not subjects and not background:
            pass
        joins = getattr(evidence, "cross_source_joins", []) or []
        if joins:
            out.append("Cross-source correlations (value seen in multiple sources):")
            for j in joins[:10]:
                out.append(f"- {j.get('value')} in {', '.join(j.get('sources', []))}")
        chrono = getattr(evidence, "chronology", []) or []
        if chrono:
            out.append("Chronology (first events):")
            for ev in chrono[:15]:
                ent = ", ".join(f"{k}={v}" for k, v in (ev.entities or {}).items())
                out.append(
                    f"- {ev.timestamp or '?'} [{ev.source}] "
                    f"{ev.actor or '-'} {ev.action or ''} {ent}".rstrip()
                )
        return out

    @staticmethod
    def _fallback_implications(ranked_anomalies, evidence, analysis, correlation=None):
        """Deterministic implications body (no LLM). Prefers the brief's window-aware assets."""
        out = []
        seen = set()
        for an in ranked_anomalies:
            imp = (getattr(an, "potential_implications", "") or "").strip()
            if imp and imp.lower() not in seen:
                seen.add(imp.lower())
                out.append(f"- {imp}")
        impact = getattr(analysis, "impact_assessment", "") if analysis else ""
        if impact and impact.strip().lower() not in seen:
            out.append(f"- Assessed impact: {impact.strip()}")
        brief = getattr(correlation, "brief", None) if correlation is not None else None
        assets = getattr(brief, "impacted_assets", None) if brief is not None else None
        assets = assets if isinstance(assets, list) else []
        in_window = [
            a
            for a in assets
            if getattr(a, "in_window", True) and getattr(a, "asset_id", "")
        ]
        if in_window:
            ids = sorted({getattr(a, "asset_id", "") for a in in_window})
            n_out = len(assets) - len(in_window)
            subj_word = ReportGenerationModule._subject_label(correlation)
            out.append(
                f"- Scope: {len(ids)} document(s) fall inside the incident window "
                f"({', '.join(ids[:8])}{'…' if len(ids) > 8 else ''}) across "
                f"{len({getattr(a, 'subject', '') for a in in_window})} {subj_word}(s)"
                + (
                    f". A further {n_out} document(s) by the same actor fall OUTSIDE the "
                    "window and are that actor's adjacent business, not incident scope."
                    if n_out
                    else "."
                )
            )
        else:
            subjects = [
                a
                for a in (getattr(evidence, "actors", []) or [])
                if getattr(a, "is_subject", False)
            ]
            per_type = {}
            for a in subjects:
                for etype, vals in (getattr(a, "entities_touched", {}) or {}).items():
                    if vals:
                        per_type.setdefault(str(etype), set()).update(vals)
            wide = {k: v for k, v in per_type.items() if len(v) > 1}
            if wide:
                bits = [
                    f"{len(v)} distinct {k}s ({', '.join(sorted(v)[:8])}"
                    f"{'…' if len(v) > 8 else ''})"
                    for k, v in sorted(wide.items(), key=lambda kv: -len(kv[1]))[:3]
                ]
                out.append(
                    f"- Scope: the person(s) of interest touched {'; '.join(bits)}, so the "
                    "impact likely extends beyond the originally alerted records."
                )
        if not out:
            out = [
                "Implications could not be derived deterministically; review the "
                "findings and evidence files to assess impact."
            ]
        return out

    @staticmethod
    def _fallback_recommendations(
        ranked_anomalies, evidence, analysis, correlation=None
    ):
        """Investigation-derived next steps (no LLM). action_backbone wins when present."""
        brief = getattr(correlation, "brief", None) if correlation is not None else None
        backbone = (
            getattr(brief, "action_backbone", None) if brief is not None else None
        )
        if isinstance(backbone, list) and backbone:
            steps = list(backbone)
            # Read off lock_target, not a label string, so gating and withholding can't drift.
            gated = bool(getattr(brief, "containment_gated", False))
            withheld = _containment_withheld(brief)
            seen = set()
            for an in ranked_anomalies[:5]:
                rec = (getattr(an, "recommended_actions", "") or "").strip()
                if rec and rec.lower() not in seen:
                    seen.add(rec.lower())
                    conf = getattr(an, "confidence_score", "?")
                    if withheld:
                        steps.append(
                            f"NOT APPLICABLE to this verdict (from anomaly, confidence "
                            f"{conf}) — the validation verdict nominated no containment "
                            "target, so this is recorded for the reviewer's judgement and "
                            f"must NOT be executed on the strength of this report: {rec}"
                        )
                    elif gated:
                        steps.append(
                            f"PROPOSED (from anomaly, confidence {conf}) — REQUIRES EXPERT "
                            "CONFIRMATION before execution; do NOT act on it "
                            f"automatically: {rec}"
                        )
                    else:
                        steps.append(
                            f"REMEDIATE (from anomaly, confidence {conf}): {rec}"
                        )
            return steps

        steps = []
        actors = [
            act
            for act in (getattr(evidence, "actors", []) or [])
            if act.actor and act.actor != "(unattributed)"
        ]
        subjects = [act for act in actors if getattr(act, "is_subject", False)]
        contain_targets = subjects or actors[:1]
        for act in contain_targets[:3]:
            ent_bits = []
            for etype, vals in (getattr(act, "entities_touched", {}) or {}).items():
                if vals:
                    ent_bits.append(f"{etype} {', '.join(vals[:3])}")
            ent_txt = f" (linked to {'; '.join(ent_bits)})" if ent_bits else ""
            steps.append(
                f"CONTAIN: review and, if confirmed, lock the responsible identity "
                f"'{act.actor}' — {act.event_count} event(s) across "
                f"{', '.join(act.sources)}{ent_txt}."
            )
        joins = getattr(evidence, "cross_source_joins", []) or []
        for j in joins[:3]:
            steps.append(
                f"INVESTIGATE: entity '{j.get('value')}' appears across "
                f"{', '.join(j.get('sources', []))} — trace its full activity for scope."
            )
        seen = set()
        for an in ranked_anomalies[:5]:
            rec = (getattr(an, "recommended_actions", "") or "").strip()
            if rec and rec.lower() not in seen:
                seen.add(rec.lower())
                conf = getattr(an, "confidence_score", "?")
                steps.append(f"REMEDIATE (from anomaly, confidence {conf}): {rec}")
        if not steps:
            return [
                "Review the retrieved evidence manually — no responsible actor, "
                "cross-source correlation, or scored anomaly could be derived."
            ]
        return steps

    @staticmethod
    def _verdict_section(correlation, phrases=None):
        """Per-subject verdict section. Returns ``None`` when no verdict was produced."""
        verdict = (
            getattr(correlation, "verdict", None) if correlation is not None else None
        )
        # subjects must be a real, non-empty list; a MagicMock attribute is truthy but
        # not a list, so guard to return None for mocked correlations without a verdict.
        subjects = getattr(verdict, "subjects", None) if verdict is not None else None
        if verdict is None or not isinstance(subjects, list) or not subjects:
            return None

        # Pack's word for the procedure; undeclared, the sentence simply omits it.
        scheme = str(getattr(verdict, "label_scheme", "") or "").upper()
        subsections = []
        intro = [
            f"Each subject below was validated against the "
            f"{scheme + ' ruleset' if scheme else 'ruleset'} "
            f"(the official procedure encoded in the knowledge pack). "
            f"Rollup — {verdict.summary or 'n/a'}.",
        ]
        if verdict.degraded:
            intro.append(
                "NOTE: some conditions could not be evaluated from the retrieved data "
                "(marked UNKNOWN below); a subject with a decisive UNKNOWN is reported as "
                "INSUFFICIENT DATA — pull the full window and re-run before deciding."
            )

        for s in verdict.subjects:
            checks = getattr(s, "checks", []) or []
            lines = [f"VERDICT: {s.verdict}"]
            decisive_note = next(
                (
                    n.split("=", 1)[1]
                    for n in (getattr(s, "notes", []) or [])
                    if str(n).startswith("decisive_exclusion=")
                ),
                "",
            )
            if decisive_note:
                lines.append(f"DECISIVE CONDITION(S): {decisive_note}")
            else:
                dec_unknown = [
                    c for c in checks if c.result == "unknown" and c.decisive
                ]
                dec_ind = [
                    c
                    for c in checks
                    if c.result == "fail"
                    and getattr(c, "polarity", "exclusion") == "fraud_indicator"
                ]
                if dec_unknown:
                    lines.append(
                        "DECISIVE CONDITION(S): none could be evaluated — "
                        + "; ".join(c.label or c.id for c in dec_unknown)
                        + " returned no usable data."
                    )
                elif dec_ind:
                    lines.append(
                        "DECISIVE CONDITION(S): positive fraud indicator(s) — "
                        + "; ".join(
                            (
                                f"{c.label or c.id} ({c.observed})"
                                if c.observed
                                else (c.label or c.id)
                            )
                            for c in dec_ind
                        )
                    )
                else:
                    passed = [c for c in checks if c.result == "pass" and c.decisive]
                    if passed:
                        lines.append(
                            "DECISIVE CONDITION(S): no exclusion fired — every decisive "
                            "check passed ("
                            + "; ".join(c.label or c.id for c in passed)
                            + "), which is the fraud fingerprint itself."
                        )
            # Surfaced only when indicators drove the verdict (no categorical exclusion outranked).
            ind_fails = [
                c
                for c in checks
                if getattr(c, "polarity", "exclusion") == "fraud_indicator"
                and getattr(c, "result", "") == "fail"
            ]
            overridden = any(
                getattr(c, "polarity", "exclusion") != "fraud_indicator"
                and getattr(c, "exclusion_kind", "heuristic") == "categorical"
                and getattr(c, "result", "") == "fail"
                for c in checks
            )
            if ind_fails and not overridden:
                names = "; ".join(
                    getattr(c, "label", "") or getattr(c, "id", "") for c in ind_fails
                )
                lines.append(
                    _phrase(
                        phrases,
                        "indicator_driven_fraud",
                        "POSITIVE FRAUD INDICATOR(S) — this verdict rests on positive "
                        "evidence of fraud: {names}. An EXPERT MUST CONFIRM before "
                        "containment (do NOT auto-void/lock/freeze); widen the search for "
                        "other assets touched by the same identity.",
                        names=names,
                    )
                )
            # asset_label= is pack-declared; engine owns the count and layout, not the noun.
            _asset_word = next(
                (
                    str(n).split("=", 1)[1]
                    for n in getattr(s, "notes", []) or []
                    if str(n).startswith("asset_label=")
                ),
                "asset",
            )
            for n in getattr(s, "notes", []) or []:
                if n.startswith("asset_count="):
                    lines.append(
                        f"{_asset_word.capitalize()}s impacted: {n.split('=', 1)[1]}"
                    )
                elif n.startswith("route="):
                    lines.append(f"In-scope route: {n.split('=', 1)[1]}")
                elif n.startswith("Multiple acting "):
                    lines.append(n)
                elif n.startswith("data_coverage="):
                    lines.append(f"Data coverage: {n.split('=', 1)[1]}")
                elif n.startswith("as_of=") or n.startswith("as_of_fallback="):
                    lines.append(f"Evidence as of the incident: {n.split('=', 1)[1]}")
                elif n.startswith("source_unanswered="):
                    lines.append(f"SOURCE DID NOT ANSWER — {n.split('=', 1)[1]}")
                elif n.startswith("procedure_unselected="):
                    lines.append(
                        f"PROCEDURE NOT SELECTED — {n.split('=', 1)[1]}"
                    )
                elif n.startswith("co_identity="):
                    lines.append(f"Identity resolution: {n.split('=', 1)[1]}")
                elif n.startswith("categorical_exclusion="):
                    lines.append(
                        _phrase(
                            phrases,
                            "categorical_exclusion",
                            "CATEGORICAL EXCLUSION — {detail}",
                            detail=n.split("=", 1)[1],
                        )
                    )
                elif n.startswith(
                    ("evidence_floor=", "no_verdict_reason=", "no_verdict_partial=")
                ):
                    lines.append(f"No verdict on the merits — {n.split('=', 1)[1]}")
            subsections.append(
                {
                    "section_title": f"{s.subject_type.upper()} {s.subject_value}",
                    "content": lines,
                }
            )

        return {
            "section_title": f"{scheme} Verdict",
            "content": intro + subsections,
        }

    @staticmethod
    def _brief_of(correlation):
        """The InvestigationBrief on a correlation result, or None. MagicMock-safe."""
        return getattr(correlation, "brief", None) if correlation is not None else None

    @classmethod
    def _declared_facts_of(cls, correlation):
        """The alert-declared facts, or ``[]``. MagicMock-safe."""
        brief = cls._brief_of(correlation)
        af = getattr(brief, "alert_facts", None) if brief is not None else None
        facts = getattr(af, "declared_facts", None) if af is not None else None
        return facts if isinstance(facts, list) else []

    @classmethod
    def _alert_facts_section(cls, correlation, phrases=None):
        """The alert's facts, verbatim. Returns ``None`` when the ruleset declares none."""
        brief = cls._brief_of(correlation)
        af = getattr(brief, "alert_facts", None) if brief is not None else None
        facts = getattr(af, "declared_facts", None) if af is not None else None
        if af is None or not isinstance(facts, list):
            return None

        lines = []
        if facts:
            lines.append(
                "These are the facts the ORIGINATING ALERT states. They are reproduced here "
                "verbatim and are authoritative: this investigation must reproduce them, and "
                "any divergence between them and the retrieved data is reported as a defect in "
                "the retrieval (see the reconciliation section) — never resolved by preferring "
                "the data."
            )
        trigger = str(getattr(af, "trigger", "") or "").strip()
        if trigger:
            lines.append(f"WHAT THE DETECTOR FIRES ON: {trigger}")
        if getattr(af, "located", False):
            lines.append(
                f"Alert record READ: {getattr(af, 'record_id', '') or '(unlabelled)'} "
                f"from {getattr(af, 'source', '') or '(unknown source)'} "
                f"({getattr(af, 'locator', '')})."
            )
        elif str(getattr(af, "locator", "") or "").strip():
            # Locator set means location was attempted and failed; empty locator means
            # no alert_record declared (nothing to assert here).
            lines.append(
                "ALERT RECORD NOT READ — "
                + str(getattr(af, "locator", ""))
                + ". Every fact below is therefore taken from the submitted alert text "
                "only, and none of it was confirmed at source."
            )
        by_field: dict = {}
        for f in facts:
            by_field.setdefault(getattr(f, "field", "?"), []).append(
                str(getattr(f, "declared", ""))
            )
        for field, values in by_field.items():
            vals = [v for v in dict.fromkeys(values) if v]
            lines.append(f"  {field}: {', '.join(vals) if vals else '(not stated)'}")
        unrelated = getattr(af, "unrelated_records", None)
        if isinstance(unrelated, list) and unrelated:
            lines.append(
                "Records in the same source belonging to a DIFFERENT incident — NOT this "
                "incident's, so no verdict, finding or action below concerns them (they may "
                "still appear in the chronology as background activity in the same window): "
                + ", ".join(str(u) for u in unrelated[: cls._UNRELATED_QUOTE_CAP])
                + "."
            )
            # An id that falls off the end is a foreign record the report stops excluding.
            note = cls._cut_note(
                len(unrelated),
                cls._UNRELATED_QUOTE_CAP,
                "record(s) of other incidents, equally excluded from every finding",
            )
            if note:
                lines.append(f"  {note}")
        if not lines:
            return None
        return {"section_title": cls._SEC_ALERT_FACTS, "content": lines}

    @classmethod
    def _reconciliation_section(cls, correlation, phrases=None):
        """Declared vs found per fact. CONFIRMED / MISMATCH / NOT FOUND / STATED."""
        brief = cls._brief_of(correlation)
        af = getattr(brief, "alert_facts", None) if brief is not None else None
        facts = getattr(af, "declared_facts", None) if af is not None else None
        if af is None or not isinstance(facts, list) or not facts:
            return None

        marks = {
            "confirmed": "CONFIRMED",
            "mismatch": "MISMATCH",
            "not_found": "NOT FOUND IN DATA",
            "stated": "STATED BY THE ALERT (no corroborating source declared)",
        }
        rows = []
        for f in facts:
            status = getattr(f, "status", "not_found")
            declared = getattr(f, "declared", "")
            found = getattr(f, "found", "")
            src = getattr(f, "source", "")
            row = (
                f"[{marks.get(status, status.upper())}] {getattr(f, 'field', '?')}: "
                f"declared '{declared}'"
            )
            if status == "confirmed":
                row += f" — found '{found}' in {src}"
            elif status == "mismatch":
                row += f" — but {src} carries '{found}'"
            elif found and src:
                # Source had other values but couldn't contradict the alert (truncated
                # or query never named this subject).
                row += f" — {src} carries '{found}', which does not settle it"
            note = getattr(f, "note", "")
            if note:
                row += f" ({note})"
            rows.append(row)

        mismatches = [f for f in facts if getattr(f, "status", "") == "mismatch"]
        gaps = [f for f in facts if getattr(f, "status", "") == "not_found"]
        lines = [
            "Every fact the alert declares, reconciled against the retrieved data.",
        ]
        lines.extend(rows)
        if mismatches:
            lines.append(
                f"DELTA — {len(mismatches)} declared fact(s) are CONTRADICTED by the data: "
                + "; ".join(
                    f"{getattr(f, 'field', '?')} (alert '{getattr(f, 'declared', '')}' vs "
                    f"{getattr(f, 'source', '')} '{getattr(f, 'found', '')}')"
                    for f in mismatches
                )
                + ". The alert is authoritative; treat each of these as a retrieval or "
                "parsing defect to investigate, not as a correction to the alert."
            )
        if gaps:
            lines.append(
                f"DELTA — {len(gaps)} declared fact(s) could not be reproduced from the "
                "retrieved data: "
                + ", ".join(
                    f"{getattr(f, 'field', '?')} '{getattr(f, 'declared', '')}'"
                    for f in gaps
                )
                + ". This is a gap in the evidence, not a contradiction of the alert."
            )
        if not mismatches and not gaps:
            lines.append(
                "DELTA: none — every declared fact that had a corroborating source was "
                "reproduced from the retrieved data."
            )
        return {"section_title": cls._SEC_RECONCILIATION, "content": lines}

    @classmethod
    def _conditions_section(cls, correlation, phrases=None):
        """Condition assessment: one block per subject, groups ordered gate → validation → hint."""
        verdict = (
            getattr(correlation, "verdict", None) if correlation is not None else None
        )
        subjects = getattr(verdict, "subjects", None) if verdict is not None else None
        if not isinstance(subjects, list) or not subjects:
            return None
        groups = (
            getattr(verdict, "condition_groups", None) if verdict is not None else None
        )
        groups = groups if isinstance(groups, list) else []
        known = ("gate", "validation", "hint")
        ordered = [
            g for role in known for g in groups if getattr(g, "role", "") == role
        ]
        ordered += [g for g in groups if getattr(g, "role", "") not in known]
        subsections = cls._condition_subsections(
            subjects, ordered, fallback_all=not groups
        )
        if not subsections:
            return None
        return {
            "section_title": cls._SEC_CONDITIONS,
            "content": [
                "PASS = the condition is satisfied. FAIL = it is violated. NOT EVALUATED = "
                "it could not be answered from the retrieved data — which is NOT a pass, and "
                "a decisive condition left NOT EVALUATED blocks a fraud verdict. '(decisive)' "
                "marks a condition that alone settles the outcome. A SCOPE group is answered "
                "before the subject is examined at all (out of scope means the procedure does "
                "not apply — not that the subject was examined and cleared); a VALIDATION "
                "group is a mandatory step; a SIGNAL group is optional and several of its "
                "entries are exculpatory, so a signal that decided a case is marked decisive "
                "like any other condition.",
            ]
            + subsections,
        }

    # Gate's PASS/FAIL = "in scope"/"out of scope", not "condition satisfied/violated".
    _ROLE_MARKS = {
        "gate": {
            "pass": "IN SCOPE",
            "fail": "OUT OF SCOPE",
            "unknown": "NOT EVALUATED",
        },
    }
    _DEFAULT_MARKS = {"pass": "PASS", "fail": "FAIL", "unknown": "NOT EVALUATED"}
    _ROLE_TAGS = {
        "gate": "SCOPE — answered before the subject is examined",
        "validation": "VALIDATION — mandatory steps",
        "hint": "SIGNALS — optional, and several are exculpatory",
    }
    #: Display caps; named so siblings cannot drift. Every item still feeds counts/determinations.
    _DERIVED_QUOTE_CAP = 10
    _NEW_SCOPE_CAP = 20
    _CHRONO_LINE_CAP = 40
    _ASSET_TABLE_CAP = 60
    _WIDER_ASSET_CAP = 40
    _UNRELATED_QUOTE_CAP = 10
    _MISSING_VALUE_QUOTE_CAP = 4
    #: Bucket for checks whose report_group matches no declared group.
    _ORPHAN_GROUP = SimpleNamespace(
        id="",
        title="Checks this ruleset filed under no declared group",
        description=(
            "These checks were evaluated and counted in the verdict above, but their "
            "`report_group` matches none of the groups the ruleset declares, so the "
            "procedure states no heading for them. Read them as ordinary checks; the "
            "grouping is a defect in the ruleset, not in the evidence."
        ),
        role="",
    )

    @classmethod
    def _cut_note(cls, total, shown, noun, compact=False):
        """Wording for every display cut; returns ``""`` when nothing was cut.

        ``compact`` is for a cut nested inside another line's parenthesis.
        """
        try:
            rest = int(total) - int(shown)
        except (TypeError, ValueError):
            return ""
        if rest <= 0:
            return ""
        if compact:
            return f"and {rest} more not listed"
        return (
            f"… and {rest} further {noun} not printed here (there are {total} in all). "
            "The cut is for length only — every one of them is included in the counts and "
            "determinations above."
        )

    @classmethod
    def _condition_subsections(cls, subjects, groups, fallback_all=False):
        """One nested block per group per subject. ``fallback_all=True`` for ungrouped rulesets."""
        out = []
        for s in subjects:
            checks = getattr(s, "checks", []) or []
            if not checks:
                continue
            if fallback_all:
                buckets = [(None, checks)]
            else:
                declared = {g.id for g in groups}
                buckets = [
                    (g, [c for c in checks if getattr(c, "group", "") == g.id])
                    for g in groups
                ]
                orphans = [
                    c for c in checks if getattr(c, "group", "") not in declared
                ]
                if orphans:
                    buckets.append((cls._ORPHAN_GROUP, orphans))
            blocks = []
            for g, group_checks in buckets:
                if not group_checks:
                    continue
                role = getattr(g, "role", "") if g is not None else ""
                marks = cls._ROLE_MARKS.get(role, cls._DEFAULT_MARKS)
                rows = []
                for c in group_checks:
                    mark = marks.get(c.result, "?")
                    star = " (decisive)" if getattr(c, "decisive", False) else ""
                    row = f"[{mark}] {c.label or c.id}{star}"
                    # Drop a value that merely repeats the mark (e.g. "NOT EVALUATED").
                    value = str(getattr(c, "observed", "") or "").strip()
                    if value and value.upper() != mark:
                        row += f" — evidence: {value}"
                    detail = str(getattr(c, "detail", "") or "").strip()
                    if detail and detail.lower() != value.lower():
                        row += (
                            f" — {detail}" if "evidence:" not in row else f" | {detail}"
                        )
                    expected = str(getattr(c, "expected", "") or "").strip()
                    if expected:
                        row += f" | required: {expected}"
                    rows.append(row)
                if role == "gate":
                    for n in getattr(s, "notes", []) or []:
                        if str(n).startswith("scope_gate="):
                            rows.append(str(n).split("=", 1)[1])
                if g is None:
                    blocks.extend(rows)
                    continue
                body = (
                    [(getattr(g, "description", "") or "").strip()]
                    if g.description
                    else []
                )
                tag = cls._ROLE_TAGS.get(role, "")
                blocks.append(
                    {
                        "section_title": (g.title or g.id)
                        + (f" [{tag}]" if tag else ""),
                        "content": body + rows,
                    }
                )
            if blocks:
                out.append(
                    {
                        "section_title": f"{s.subject_type.upper()} {s.subject_value}",
                        "content": blocks,
                    }
                )
        return out

    @classmethod
    def _wider_evidence_section(cls, correlation, phrases=None):
        """Wider-evidence sweep section, framed as evidence gathering rather than a finding."""
        brief = cls._brief_of(correlation)
        if brief is None:
            return None
        status = getattr(brief, "scope_status", "")
        assets = getattr(brief, "impacted_assets", None)
        links = getattr(brief, "subject_links", None)
        extra = getattr(brief, "additional_subjects", None)
        if not isinstance(status, str) or not status.strip():
            return None
        assets = assets if isinstance(assets, list) else []
        links = links if isinstance(links, list) else []
        extra = extra if isinstance(extra, list) else []

        lines = [
            _phrase(
                phrases,
                "wider_evidence_framing",
                "This is EVIDENCE GATHERING, not a finding. The sweep widens the search "
                "beyond the assets the alert named, for the same acting identity, so the "
                "evidence base is complete; further activity by the same identity is not "
                "itself an anomaly.",
            ),
            f"Sweep outcome: {status}",
        ]
        derived = [link for link in links if getattr(link, "related_subject", "")]
        if derived:
            lines.append(
                "Derivation links found in the data — these related subjects are the SAME "
                "episode under a second identifier, NOT scope the alert failed to name:"
            )
            for link in derived[: cls._DERIVED_QUOTE_CAP]:
                quote = getattr(link, "quote", "")
                role = getattr(link, "role", "unknown")
                rel = (
                    "child"
                    if role == "parent"
                    else ("parent" if role == "child" else "related")
                )
                lines.append(
                    f"  {getattr(link, 'subject', '')} → {getattr(link, 'related_subject', '')}"
                    f" ({getattr(link, 'kind', 'link')}, the {rel})"
                    + (f": {quote}" if quote else "")
                    + (
                        f" [element {getattr(link, 'element_id', '')}]"
                        if getattr(link, "element_id", "")
                        else ""
                    )
                )
            note = cls._cut_note(
                len(derived), cls._DERIVED_QUOTE_CAP, "derivation link(s) of the same kind"
            )
            if note:
                lines.append(f"  {note}")
        derived_values = {getattr(link, "related_subject", "") for link in derived}
        new_scope = [str(s) for s in extra if str(s) and str(s) not in derived_values]
        if new_scope:
            lines.append(
                "Subjects the sweep returned that the alert did NOT name and that are not "
                "derived from an alerted subject: "
                + ", ".join(new_scope[: cls._NEW_SCOPE_CAP])
                + "."
            )
            note = cls._cut_note(
                len(new_scope), cls._NEW_SCOPE_CAP, "such subject(s)"
            )
            if note:
                lines.append(f"  {note}")
        in_window = [
            a
            for a in assets
            if getattr(a, "in_window", True) and getattr(a, "asset_id", "")
        ]
        if in_window:
            lines.append(
                "Assets inside the incident window (asset | subject | amount | "
                "status | issued | by):"
            )
            for a in in_window[: cls._WIDER_ASSET_CAP]:
                amount = str(getattr(a, "amount", "") or "").strip()
                currency = str(getattr(a, "currency", "") or "").strip()
                # An amount without currency is not a monetary fact; annotate it.
                money = (
                    f"{amount} {currency}"
                    if amount and currency
                    else (
                        f"{amount} (CURRENCY NOT RETURNED)" if amount else "(no amount)"
                    )
                )
                lines.append(
                    f"  {getattr(a, 'asset_id', '')} | {getattr(a, 'subject', '')} | "
                    f"{money} | {getattr(a, 'status', '') or '?'} | "
                    f"{getattr(a, 'event_date', '') or 'issue date unknown'} | "
                    f"{getattr(a, 'actor', '') or '?'}"
                )
            note = cls._cut_note(
                len(in_window), cls._WIDER_ASSET_CAP, "asset(s) inside the window"
            )
            if note:
                lines.append(f"  {note}")
        out_window = [a for a in assets if not getattr(a, "in_window", True)]
        if out_window:
            lines.append(
                f"A further {len(out_window)} asset(s) by the same actor fall OUTSIDE the "
                "incident window. They are that actor's adjacent business: reported for "
                "review, NOT counted in this incident's scope or exposure."
            )
        return {"section_title": cls._SEC_WIDER, "content": lines}

    @classmethod
    def _not_retrieved_section(cls, correlation, phrases=None, logs=None):
        """Explicit gap list: unevaluated conditions, empty sources, truncated sources. ``None`` when clean."""
        verdict = (
            getattr(correlation, "verdict", None) if correlation is not None else None
        )
        subjects = getattr(verdict, "subjects", None) if verdict is not None else None
        lines = []
        if isinstance(subjects, list):
            for s in subjects:
                unevaluated = [
                    c for c in (getattr(s, "checks", []) or []) if c.result == "unknown"
                ]
                if not unevaluated:
                    continue
                # kind:stub = no data path; counted separately so the heading is honest.
                unwired = [
                    c
                    for c in unevaluated
                    if str(getattr(c, "observed", "") or "") == STUB_OBSERVED
                ]
                split = (
                    f" ({len(unevaluated) - len(unwired)} for missing data, "
                    f"{len(unwired)} declared with no data path)"
                    if unwired
                    else ""
                )
                lines.append(
                    f"{s.subject_type.upper()} {s.subject_value} — "
                    f"{len(unevaluated)} condition(s) NOT EVALUATED{split}:"
                )
                for c in unevaluated:
                    blocking = (
                        " [DECISIVE — this gap alone blocks a fraud verdict]"
                        if getattr(c, "decisive", False)
                        else ""
                    )
                    if str(getattr(c, "observed", "") or "") == STUB_OBSERVED:
                        blocking += (
                            " [NO DATA PATH — the procedure requires this check and none "
                            "is wired, so a re-run cannot answer it]"
                        )
                    why = (getattr(c, "detail", "") or "no data returned").strip()
                    lines.append(f"  {c.label or c.id}{blocking}: {why}")
        evidence = (
            getattr(correlation, "evidence", None) if correlation is not None else None
        )
        sources = getattr(evidence, "sources", None) if evidence is not None else None
        if isinstance(sources, list):
            # zero_rows_meaning means empty IS the answer, not a gap.
            empty = [
                s
                for s in sources
                if getattr(s, "record_count", 0) == 0
                and not str(getattr(s, "zero_rows_meaning", "") or "").strip()
            ]
            answered = [
                s
                for s in sources
                if getattr(s, "record_count", 0) == 0
                and str(getattr(s, "zero_rows_meaning", "") or "").strip()
            ]
            capped = [s for s in sources if getattr(s, "row_limited", False)]
            if empty:
                lines.append(
                    "Sources that returned NO rows: "
                    + ", ".join(str(getattr(s, "source", "")) for s in empty)
                    + ". Any condition depending only on these is unevaluated, not clear."
                )
            if answered:
                # "empty IS the answer" must not read as "not consulted at all".
                lines.append(
                    "Sources that returned NO rows where EMPTY IS THE ANSWER, not a gap — "
                    "the condition reading each is answered, not unevaluated: "
                    + "; ".join(
                        f"{getattr(s, 'source', '')} "
                        f"({str(getattr(s, 'zero_rows_meaning', '')).strip()})"
                        for s in answered
                    )
                    + "."
                )
            if capped:
                lines.append(
                    "Sources TRUNCATED at their configured row cap — every count from these "
                    "is a LOWER BOUND, never a total, and no figure in this report is "
                    "derived from one: "
                    + ", ".join(
                        f"{getattr(s, 'source', '')} (>= {getattr(s, 'record_count', 0)} rows)"
                        for s in capped
                    )
                    + "."
                )
        confirmed_pairs = {
            (str(getattr(f, "source", "") or ""), str(getattr(f, "field", "") or ""))
            for f in cls._declared_facts_of(correlation)
            if str(getattr(f, "status", "")) == "confirmed"
        }
        unreconciled = {}
        for f in cls._declared_facts_of(correlation):
            status = str(getattr(f, "status", ""))
            src = str(getattr(f, "source", "") or "")
            if not src or status not in ("mismatch", "not_found"):
                continue
            if (src, str(getattr(f, "field", "") or "")) in confirmed_pairs:
                continue
            unreconciled.setdefault(src, []).append(
                f"{getattr(f, 'field', '?')} '{getattr(f, 'declared', '')}'"
            )
        if unreconciled:
            per_source = []
            for src, vals in unreconciled.items():
                shown = ", ".join(vals[: cls._MISSING_VALUE_QUOTE_CAP])
                cut = cls._cut_note(
                    len(vals), cls._MISSING_VALUE_QUOTE_CAP, "", compact=True
                )
                per_source.append(f"{src} (missing {shown}{' ' + cut if cut else ''})")
            lines.append(
                "Sources that answered rows but do NOT carry a value the ALERT "
                "declares — read every finding derived from one as possibly not about "
                "the alerted subject at all (a stale or partially-ingested read, a "
                "wrong record, or a query that never named this subject): "
                + "; ".join(per_source)
                + ". The reconciliation section states each one and why."
            )
        brief = cls._brief_of(correlation)
        carve_outs = getattr(brief, "unenforced_carve_outs", None) if brief else None
        if isinstance(carve_outs, list) and carve_outs:
            lines.append(
                "Procedure rules this engine does NOT apply — each names a check that "
                "therefore reads more broadly than the procedure does. Not a retrieval "
                "gap: no re-run can close one."
            )
            for c in carve_outs:
                if str(c).strip():
                    lines.append(f"  {str(c).strip()}")
        if not lines:
            return None
        return {
            "section_title": cls._SEC_NOT_RETRIEVED,
            "content": [
                "Listed explicitly so every unevaluated condition is auditable rather than "
                "silently absent.",
            ]
            + lines,
        }

    @classmethod
    def _note_subthreshold(cls, sections, anomalies):
        """Record sub-threshold anomalies in the coverage section. Mutates ``sections`` in place."""
        below = [
            a for a in (anomalies or []) if bool(getattr(a, "below_threshold", False))
        ]
        if not below:
            return
        scores = sorted(
            (float(getattr(a, "confidence_score", 0.0)) for a in below), reverse=True
        )
        lines = [
            f"{len(below)} detected anomal{'y' if len(below) == 1 else 'ies'} scored "
            "BELOW the confidence cut-off and are therefore NOT part of the narrative "
            f"above (highest {scores[0]:.2f}). They are retained in full in the exported "
            "anomaly list and the evidence artifacts, and a reviewer who disagrees with "
            "the cut-off should read them there:"
        ]
        for a in sorted(
            below,
            key=lambda x: float(getattr(x, "confidence_score", 0.0)),
            reverse=True,
        ):
            desc = str(getattr(a, "description", "") or "").strip()
            if len(desc) > 300:
                desc = desc[:297].rstrip() + "..."
            lines.append(f"  [{float(getattr(a, 'confidence_score', 0.0)):.2f}] {desc}")
        for section in sections:
            if (
                isinstance(section, dict)
                and section.get("section_title") == cls._SEC_NOT_RETRIEVED
            ):
                content = section.get("content")
                if isinstance(content, list):
                    content.extend(lines)
                    return
        sections.append({"section_title": cls._SEC_NOT_RETRIEVED, "content": lines})

    @classmethod
    def _actions_section(cls, correlation, phrases=None):
        """Actions section. Read off lock_target, not label strings; withheld verdicts say so explicitly."""
        verdict = (
            getattr(correlation, "verdict", None) if correlation is not None else None
        )
        subjects = getattr(verdict, "subjects", None) if verdict is not None else None
        if verdict is None or not isinstance(subjects, list) or not subjects:
            return None

        blocks = []
        acted = False
        for s in subjects:
            lt = getattr(s, "lock_target", None)
            lt = lt if isinstance(lt, dict) else {}
            lines = []
            if lt.get("scope") or lt.get("identity"):
                acted = True
                o_label = str(lt.get("scope_label", "") or "").strip() or "scope"
                s_label = str(lt.get("identity_label", "") or "").strip() or "identity"
                prov = "; ".join(
                    f"{role} from {lt[k]}"
                    for k, role in (
                        ("scope_field", o_label.lower()),
                        ("identity_field", s_label.lower()),
                    )
                    if lt.get(k)
                )
                target = " ".join(
                    f"{lab.lower()}={lt.get(key, '')}"
                    for key, lab in (("scope", o_label), ("identity", s_label))
                    if lt.get(key)
                ) + (
                    (
                        f" (source {lt.get('source')}"
                        + (f"; {prov}" if prov else "")
                        + ")"
                    )
                    if lt.get("source")
                    else ""
                )
                lines.append(
                    _phrase(
                        phrases,
                        "containment_target",
                        "CONTAINMENT TARGET — {target}.",
                        target=target,
                    )
                )
                # Prerequisites before the action they block.
                if lt.get("prerequisites"):
                    lines.append(str(lt["prerequisites"]))
                if lt.get("action"):
                    lines.append(
                        _phrase(
                            phrases,
                            "containment_action",
                            "ACTION: {action}",
                            action=lt["action"],
                        )
                    )
                    if lt.get("action_rationale"):
                        lines.append(f"  Why: {lt['action_rationale']}")
                else:
                    # lock_target without an action = platform_mode declared but undetermined.
                    lines.append(
                        _phrase(
                            phrases,
                            "containment_action_unknown",
                            (
                                "ACTION: NOT DETERMINED — the platform this identity sells on "
                                "could not be established, and the action sets are not "
                                "interchangeable between platforms. A human must select one."
                                if lt.get("platform") == "unknown"
                                else "ACTION: NOT DETERMINED — this procedure resolves no "
                                "containment action, so a human must select one."
                            ),
                        )
                    )
                if lt.get("identity_class"):
                    lines.append(
                        f"  {s_label.capitalize()} class: {lt['identity_class']}"
                    )
            else:
                withheld = next(
                    (
                        str(n).split("=", 1)[1]
                        for n in (getattr(s, "notes", []) or [])
                        if str(n).startswith("containment_withheld=")
                    ),
                    "",
                )
                lines.append(
                    f"NO ACTION IS AUTHORISED by the '{s.verdict}' verdict on this subject: "
                    "no containment, no suspension, no void instruction and no external "
                    "notification."
                )
                if withheld:
                    lines.append(withheld)
            for n in getattr(s, "notes", []) or []:
                if str(n).startswith("platform_mode="):
                    lines.append(
                        _phrase(
                            phrases,
                            "selling_platform",
                            "Platform: {platform}",
                            platform=str(n).split("=", 1)[1],
                        )
                    )
            blocks.append(
                {
                    "section_title": f"{s.subject_type.upper()} {s.subject_value}",
                    "content": lines,
                }
            )

        intro = [
            (
                "Actions are emitted only where the verdict authorises them. Where it does not, "
                "that is stated per subject rather than left out."
                if acted
                else "This verdict authorises NO action. Nothing below is an instruction to act."
            )
        ]
        # AFIR never sends the notification draft; the heading says so explicitly.
        if getattr(verdict, "notification_draft", ""):
            blocks.append(
                {
                    "section_title": (
                        _phrase(
                            phrases,
                            "notification_draft_heading",
                            "Draft notification — NOT SENT by AFIR; for human review",
                        )
                        if acted
                        else _phrase(
                            phrases,
                            "notification_closure_heading",
                            "Draft case-closure note — NOT SENT by AFIR; no containment "
                            "or escalation is recommended",
                        )
                    ),
                    "content": [verdict.notification_draft],
                }
            )
        return {"section_title": cls._SEC_ACTIONS, "content": intro + blocks}

    # --- Advisory lane: cross-procedure correlation. State names asserted in test_report_generation.
    _LINK_STATE_BLOCKS = (
        (
            "probed_positive",
            "PRESENT IN THIS EVIDENCE",
            "the other procedure's own applicability test or declared entry signal is "
            "satisfied by rows this run already retrieved. That is a referral, not a second "
            "verdict: only a run of that procedure can adjudicate it.",
        ),
        (
            "not_probed",
            "REACHABLE BUT NOT SETTLED",
            "a value that would open the other procedure's leg is in hand, and this run did "
            "not answer the question. NOTHING below was ruled out — settling one needs a "
            "query this run did not make.",
        ),
        (
            "probed_negative",
            "CONSIDERED AND RULED OUT",
            "checked against this run's own rows and excluded. Stated rather than omitted, "
            "because a procedure that was checked and a procedure nobody asked about are "
            "otherwise the same silence, and only one of them still needs doing.",
        ),
        (
            "unreachable",
            "NOT REACHABLE FROM THIS EVIDENCE",
            "nothing in this knowledge pack connects what this run holds to the value that "
            "would open the other procedure's leg. A gap in the data model rather than in "
            "this investigation: no re-run of anything closes one.",
        ),
    )

    _LINK_PREAMBLE = (
        "ADVISORY, and addressed to a human. Nothing in this section was read by any "
        "condition, and none of it moved the verdict above, its severity, or this run's "
        "stage health — the procedure named in the verdict is the only one this run "
        "adjudicated. What follows is which OTHER procedures this evidence may reach, how "
        "far the engine got with each for free, and what it concluded. A candidate is a "
        "question worth asking next; it is not a finding about it."
    )

    #: Max pivot values quoted per candidate before the count stands in for the rest.
    _LINK_MAX_PIVOT_VALUES = 6

    #: Max score-reason terms printed beside the link score.
    _LINK_MAX_SCORE_REASONS = 4

    @staticmethod
    def _links_of(correlation):
        """Assessed link candidates, or ``[]``. Falls back to ``brief.links``; MagicMock-safe."""
        for holder in (
            correlation,
            (getattr(correlation, "brief", None) if correlation is not None else None),
        ):
            links = getattr(holder, "links", None) if holder is not None else None
            if isinstance(links, list):
                real = [f for f in links if isinstance(getattr(f, "state", None), str)]
                if real:
                    return real
        return []

    @classmethod
    def _links_section(cls, correlation, phrases=None):
        """Advisory cross-procedure section, or ``None`` when nothing was assessed."""
        findings = cls._links_of(correlation)
        if not findings:
            return None
        by_state = {}
        for finding in findings:
            by_state.setdefault(str(getattr(finding, "state", "") or ""), []).append(
                finding
            )
        known = {state for state, _, _ in cls._LINK_STATE_BLOCKS}
        blocks = [b for b in cls._LINK_STATE_BLOCKS if b[0] in by_state]
        # An unknown state prints under its own name rather than being silently dropped.
        blocks += [
            (state, (state or "unspecified").replace("_", " ").upper(), "")
            for state in sorted(by_state)
            if state not in known
        ]
        lines = []
        for state, heading, meaning in blocks:
            group = by_state.get(state) or []
            lines.append(
                f"{heading} ({len(group)}){' — ' + meaning if meaning else ''}"
            )
            for finding in group:
                lines.extend(cls._link_lines(finding))
        return {
            "section_title": cls._SEC_LINKS,
            "content": [cls._LINK_PREAMBLE]
            + cls._link_escalation_lines(findings)
            + lines,
        }

    @classmethod
    def _link_escalation_lines(cls, findings):
        """Escalation rollup, stated once even when nothing acted autonomously."""
        total = len(findings)
        escalating = [
            f
            for f in findings
            if str(getattr(f, "mode", "") or "").strip() in ESCALATING_MODES
        ]
        # `planned` from a default and `planned` from a clamp look the same off `mode`.
        clamped = [
            f
            for f in findings
            if str(getattr(f, "mode_source", "") or "").strip() == "clamp"
        ]
        if escalating:
            targets = ", ".join(
                sorted(
                    {
                        str(getattr(f, "target_use_case", "") or "").strip()
                        or "(procedure not named)"
                        for f in escalating
                    }
                )
            )
            out = [
                f"ESCALATION — {len(escalating)} of these {total} candidate(s) are set to "
                f"act without being asked ({targets}); each says below what its setting "
                "would do. Every other candidate here composes a referral for a human to "
                "execute and spent nothing."
            ]
        else:
            out = [
                f"ESCALATION — none of these {total} candidate(s) acted on its own. Every "
                "one composes a referral for a human to execute, so nothing in this section "
                "was retrieved or run without being asked for."
            ]
        if clamped:
            out.append(
                f"An escalating setting was asked for on {len(clamped)} of them and "
                "REFUSED, because the target procedure's own applicability test did not hold "
                "against the rows this run retrieved — so nothing here establishes that "
                "procedure applies, whatever the setting asked for. Those are held at a "
                "composed referral, and each names which outcome refused it below."
            )
        withheld = [
            f
            for f in findings
            if str(getattr(f, "mode_source", "") or "").strip() == "score"
        ]
        if withheld:
            out.append(
                f"On {len(withheld)} of them the setting is 'semi_auto' and this incident's "
                "score fell below the configured threshold, so they composed a referral rather "
                "than acting — that is the setting behaving as defined, not a refusal, and each "
                "states its score and the threshold below."
            )
        held = [f for f in findings if cls._mode_held_by_declaration(f)]
        if held:
            out.append(
                f"On {len(held)} of them an escalating setting was asked for and a narrower "
                "declaration holds them at a composed referral anyway — a deliberate hold rather "
                "than a refusal or a score, so nothing here failed and each names below which "
                "layer holds it."
            )
        return out

    @classmethod
    def _link_lines(cls, finding):
        """One candidate: headline plus indented provenance. Every field is optional."""
        target = (
            str(getattr(finding, "target_use_case", "") or "").strip()
            or "(procedure not named)"
        )
        direction = str(getattr(finding, "direction", "") or "").strip()
        head = f"  -> {target}" + (f" [{direction}]" if direction else "")
        entity = str(getattr(finding, "pivot_entity", "") or "").strip()
        raw_values = getattr(finding, "pivot_values", None)
        values = (
            [str(v).strip() for v in raw_values if str(v).strip()]
            if isinstance(raw_values, list)
            else []
        )
        if entity and values:
            shown = values[: cls._LINK_MAX_PIVOT_VALUES]
            cut = cls._cut_note(len(values), len(shown), "value", compact=True)
            head += f" — via {entity} {', '.join(shown)}" + (f" ({cut})" if cut else "")
        elif entity:
            head += f" — would need a {entity} value, and this run holds none"
        out = [head]
        note = str(getattr(finding, "evidence_note", "") or "").strip()
        if note:
            out.append(f"      what this run's rows show: {note}")
        gap = str(getattr(finding, "gap_reason", "") or "").strip()
        if gap:
            out.append(f"      why the assessment stopped there: {gap}")
        window = str(getattr(finding, "window_hint", "") or "").strip()
        if window:
            out.append(f"      a referral should ask over: {window}")
        signal = str(getattr(finding, "signal_id", "") or "").strip()
        rung = getattr(finding, "rung", None)
        basis = f"declared entry signal '{signal}'" if signal else "no entry signal"
        if isinstance(rung, int):
            basis += f", assessed to rung {rung}"
        out.append(f"      basis: {basis}")
        out.extend(cls._link_score_lines(finding))
        rate = str(getattr(finding, "base_rate", "") or "").strip()
        if rate:
            out.append(f"      the pack's measured base rate for it: {rate}")
        elif signal:
            out.append(
                "      the pack shipped this signal UNMEASURED, so that it fired is "
                "reported and not relied on"
            )
        out.extend(cls._link_mode_lines(finding))
        severity = str(getattr(finding, "advisory_severity", "") or "").strip()
        if severity:
            advisory = str(getattr(finding, "advisory_note", "") or "").strip()
            out.append(
                f"      advisory severity {severity} — router-added, addressed to a human, "
                "and not this run's severity" + (f": {advisory}" if advisory else "")
            )
        return out

    @classmethod
    def _link_score_lines(cls, finding):
        """The candidate's score and the terms that produced it, or ``[]`` when absent."""
        raw = getattr(finding, "link_score", None)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return []
        score = max(0.0, min(1.0, float(raw)))
        line = f"      link score: {score:.2f} of 1.00 (deterministic, from the free rungs only)"
        raw_reasons = getattr(finding, "link_score_reasons", None)
        reasons = (
            [str(r).strip() for r in raw_reasons if str(r).strip()]
            if isinstance(raw_reasons, list)
            else []
        )
        if score > 0 and reasons:
            shown = reasons[: cls._LINK_MAX_SCORE_REASONS]
            cut = cls._cut_note(len(reasons), len(shown), "term", compact=True)
            line += " — " + "; ".join(shown) + (f" ({cut})" if cut else "")
        return [line]

    @staticmethod
    def _mode_held_by_declaration(finding):
        """True when a narrower layer's declaration holds the mode, not a clamp or score gate."""
        mode = str(getattr(finding, "mode", "") or "").strip()
        source = str(getattr(finding, "mode_source", "") or "").strip()
        note = str(getattr(finding, "mode_note", "") or "").strip()
        return bool(
            mode
            and mode not in ESCALATING_MODES
            and source not in ("clamp", "score")
            and note
        )

    @classmethod
    def _link_mode_lines(cls, finding):
        """This candidate's escalation setting, silent when at the default."""
        mode = str(getattr(finding, "mode", "") or "").strip()
        source = str(getattr(finding, "mode_source", "") or "").strip()
        action = str(getattr(finding, "proposed_action", "") or "").strip()
        note = str(getattr(finding, "mode_note", "") or "").strip()
        if source == "clamp":
            out = [
                "      escalation: an automatic setting was asked for here and REFUSED, so "
                "this candidate only proposes"
            ]
            if note:
                out.append(f"      why it was refused: {note}")
            return out
        if source == "score":
            out = [
                "      escalation: this pair is set to act on its own above a score, and this "
                "candidate's evidence did not reach it, so it composes a referral instead"
            ]
            if note:
                out.append(f"      the arithmetic: {note}")
            return out
        if mode not in ESCALATING_MODES:
            if not cls._mode_held_by_declaration(finding):
                return []
            return [
                f"      escalation: this pair is held at '{mode}'"
                + (f" by the {source} setting" if source else "")
                + ", overruling a wider setting that asked for more, so it only proposes",
                f"      why it is held: {note}",
            ]
        out = [
            f"      escalation: this pair is set to '{mode}'"
            + (f", from the {source} setting" if source else "")
            + (f" — it will {action}" if action else "")
        ]
        if note:
            out.append(f"      about that setting: {note}")
        return out

    _INQUIRY_STATE_BLOCKS = (
        (
            "answered",
            "ASKED AND ANSWERED",
            "one further reading of the named source returned rows, and the procedure "
            "declared in advance what those rows mean. A reading, not an adjudication: no "
            "condition above consumed any of it.",
        ),
        (
            "not_asked",
            "STILL OPEN",
            "settling one of these needs a query this run did not make. Nothing below was "
            "ruled out, and nothing below is a finding — each is a question with a named "
            "source to put it to.",
        ),
        (
            "empty",
            "ASKED AND THE SOURCE HAD NOTHING",
            "the source answered with no matching rows, and the procedure declared what that "
            "means here. Stated rather than omitted, because an empty answer and a question "
            "nobody asked are otherwise the same silence.",
        ),
        (
            "unanswered",
            "ASKED AND THE SOURCE DID NOT ANSWER",
            "the query was made and the source returned nothing at all — a timeout, a "
            "credential or a catalog gap. That is not an empty answer: the remedy is the "
            "environment, and re-reading these rows cannot supply one.",
        ),
        (
            "unreachable",
            "NOT ASKABLE FROM THIS EVIDENCE",
            "the question is scoped by a value this run does not hold, so any query would "
            "scan the source over the whole window instead of asking about one identity. A "
            "gap in the declaration or in the data model, not in this investigation.",
        ),
    )

    _INQUIRY_PREAMBLE = (
        "ADVISORY, and addressed to a human. These are questions the adjudicating procedure "
        "declared about its OWN evidence — what this run could not settle. Nothing in this "
        "section was read by any condition, and none of it moved the verdict above, its "
        "severity, or this run's stage health. An answer here is a reading of rows against a "
        "meaning the procedure wrote down in advance; it is not a determination, and a "
        "question that stayed open is not a negative finding."
    )

    #: Max scope values quoted per question before the count stands in for the rest.
    _INQUIRY_MAX_SCOPE_VALUES = 6

    @staticmethod
    def _inquiries_of(correlation):
        """Assessed open questions, or ``[]``. Falls back to ``brief.inquiries``; MagicMock-safe."""
        for holder in (
            correlation,
            (getattr(correlation, "brief", None) if correlation is not None else None),
        ):
            found = getattr(holder, "inquiries", None) if holder is not None else None
            if isinstance(found, list):
                real = [f for f in found if isinstance(getattr(f, "state", None), str)]
                if real:
                    return real
        return []

    @classmethod
    def _inquiries_section(cls, correlation, phrases=None):
        """Advisory open-question section, or ``None`` when the pack declared none."""
        findings = cls._inquiries_of(correlation)
        if not findings:
            return None
        by_state = {}
        for finding in findings:
            by_state.setdefault(str(getattr(finding, "state", "") or ""), []).append(
                finding
            )
        known = {state for state, _, _ in cls._INQUIRY_STATE_BLOCKS}
        blocks = [b for b in cls._INQUIRY_STATE_BLOCKS if b[0] in by_state]
        # An unknown state prints under its own name rather than being silently dropped.
        blocks += [
            (state, (state or "unspecified").replace("_", " ").upper(), "")
            for state in sorted(by_state)
            if state not in known
        ]
        lines = []
        for state, heading, meaning in blocks:
            group = by_state.get(state) or []
            lines.append(
                f"{heading} ({len(group)}){' — ' + meaning if meaning else ''}"
            )
            for finding in group:
                lines.extend(cls._inquiry_lines(finding))
        return {
            "section_title": cls._SEC_INQUIRIES,
            "content": [cls._INQUIRY_PREAMBLE]
            + cls._inquiry_cost_lines(findings)
            + lines,
        }

    @classmethod
    def _inquiry_cost_lines(cls, findings):
        """What this lane spent, stated once even when it spent nothing."""
        total = len(findings)
        spent = [f for f in findings if bool(getattr(f, "probe_spent", False))]
        free = [
            f
            for f in findings
            if not bool(getattr(f, "probe_spent", False))
            and str(getattr(f, "state", "") or "")
            in ("answered", "empty", "unanswered")
        ]
        if spent:
            out = [
                f"COST — {len(spent)} of these {total} question(s) cost one bounded query "
                "each, made after the verdict was already decided and outside every source "
                "the conditions read. Each says below what it asked and what came back."
            ]
        else:
            out = [
                f"COST — none of these {total} question(s) cost a query. Each was either "
                "settled from rows this run had already retrieved, or is reported unsettled "
                "with the reason nothing was spent on it."
            ]
        if free:
            out.append(
                f"{len(free)} of them were settled at no retrieval cost, by re-reading rows "
                "this run had already retrieved for its own conditions — a bounded re-use of "
                "evidence in hand, and each says so."
            )
        capped = [f for f in findings if bool(getattr(f, "row_cap_hit", False))]
        if capped:
            out.append(
                f"On {len(capped)} of them the rows read were CAPPED, so the count below is a "
                "floor and not a total — a larger answer would have been truncated the same "
                "way, and neither reading changes anything above."
            )
        return out

    @classmethod
    def _inquiry_lines(cls, finding):
        """One open question: headline plus indented provenance. Every field is optional."""
        question = str(getattr(finding, "question", "") or "").strip()
        qid = str(getattr(finding, "id", "") or "").strip()
        head = f"  ?> {question or qid or '(question not stated)'}"
        if question and qid:
            head += f" [{qid}]"
        out = [head]
        entity = str(getattr(finding, "scope_entity", "") or "").strip()
        raw_values = getattr(finding, "scope_values", None)
        values = (
            [str(v).strip() for v in raw_values if str(v).strip()]
            if isinstance(raw_values, list)
            else []
        )
        if entity and values:
            shown = values[: cls._INQUIRY_MAX_SCOPE_VALUES]
            cut = cls._cut_note(len(values), len(shown), "value", compact=True)
            out.append(
                f"      asked about {entity} {', '.join(shown)}"
                + (f" ({cut})" if cut else "")
            )
        elif entity:
            out.append(
                f"      would be asked about a {entity} value, and this run holds none"
            )
        source = str(getattr(finding, "source", "") or "").strip()
        if source:
            out.append(f"      the source that would answer it: {source}")
        trigger = str(getattr(finding, "trigger", "") or "").strip()
        if trigger:
            out.append(f"      why it was raised: {trigger}")
        rows = getattr(finding, "rows_matched", None)
        if isinstance(rows, int) and not isinstance(rows, bool):
            floor = " (a floor — the rows read were capped)" if getattr(
                finding, "row_cap_hit", False
            ) else ""
            out.append(f"      rows matching it: {rows}{floor}")
        meaning = str(getattr(finding, "meaning", "") or "").strip()
        if meaning:
            out.append(f"      what the procedure says that means: {meaning}")
        gap = str(getattr(finding, "gap_reason", "") or "").strip()
        if gap:
            out.append(f"      why it is not settled: {gap}")
        probe = str(getattr(finding, "probe_note", "") or "").strip()
        if probe:
            out.append(f"      what it cost: {probe}")
        note = str(getattr(finding, "note", "") or "").strip()
        if note:
            out.append(f"      the procedure's own note on it: {note}")
        advisory = str(getattr(finding, "advisory_note", "") or "").strip()
        if advisory:
            out.append(f"      provenance: {advisory}")
        return out

    @staticmethod
    def _evidence_artifacts_section(incident_id):
        """Deterministic 'Evidence Artifacts' section describing the two evidence files."""
        raw = f"evidence_raw_{incident_id}.json"
        transformed = f"evidence_transformed_{incident_id}.json"
        return {
            "section_title": "Evidence Artifacts",
            "content": [
                "Two evidence files are saved next to this report (in the exports "
                "directory). Every claim above can be traced back to them. The report "
                "stays a readable narrative; the bulk data lives in these files.",
                {
                    "section_title": raw,
                    "content": [
                        "WHAT: the complete raw dataset — every log record returned by "
                        "every source that was queried, exactly as extracted (no "
                        "trimming, no aggregation).",
                        "HOW GENERATED: the log-retrieval stage ran one LLM-generated "
                        "backend query per source (ES|QL / SQL / DSL) scoped to the "
                        "incident's entities and time window; each source's rows are "
                        "stored verbatim under its source name.",
                        "HOW TO USE: it is a JSON object keyed by source name, exactly "
                        "as the sources are named in the findings above; each value is the "
                        "list of that source's rows. Open the source named in a finding, "
                        "then filter its rows by the entity value cited in that finding to "
                        "see the exact underlying events and every available field.",
                    ],
                },
                {
                    "section_title": transformed,
                    "content": [
                        "WHAT: the transformed / analytical view — what the correlation "
                        "stage computed from the raw data.",
                        "HOW GENERATED: after retrieval, the correlation stage "
                        "deterministically parsed every row (any shape — nested, JSON, "
                        "variant), resolved which fields join across sources, computed "
                        "cross-source overlaps within the time window, and built a merged "
                        "chronology + per-actor attribution. No data was invented.",
                        "HOW TO USE: keys to look at — 'evidence.chronology' (the merged "
                        "time-ordered events, subject events first), 'evidence.actors' "
                        "(per-identity rollup with is_subject marking the persons of "
                        "interest), 'evidence.cross_source_joins' (values seen in more "
                        "than one source), and 'aggregations.resolved_correlation_keys' "
                        "(which fields were joined on). Use this file to see the shape of "
                        "the incident; use the raw file to drill into individual events.",
                    ],
                },
            ],
        }

    def generate_pdf_report(self, sections, incident, anomalies):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Justify", alignment=1))

        story = []

        if self.config.get("logo_path") and os.path.exists(self.config["logo_path"]):
            logo = Image(self.config["logo_path"], width=250, height=40)
            story.append(logo)
            story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"Fraud Investigation Report - Incident {incident['id']}",
                styles["Title"],
            )
        )
        story.append(Spacer(1, 12))

        for section in sections:
            story = self.append_pdf_content(story, section, styles)
            story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def export_report(self, report, file_type, path):
        # Anchor the output dir: use the configured path if it exists, else fall
        # back to the repo/AFIR_DATA_DIR exports dir so it works from any cwd.
        out_dir = path if path and os.path.isdir(path) else str(exports_dir())
        if file_type == "pdf":
            with open(os.path.join(out_dir, "fraud_report.pdf"), "wb") as f:
                f.write(report)
            logger.info("Report generated and saved")
        else:
            with open(os.path.join(out_dir, "fraud_report.txt"), "w") as f:
                f.write(report)
