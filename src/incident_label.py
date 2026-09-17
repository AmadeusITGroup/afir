"""Fixed-width human names for a run: the id stays the key, the label carries the meaning.

An incident ``id`` cannot be the meaningful one. It names artifacts
(``fraud_report_<id>.pdf``, ``evidence_raw_<id>.json``), it is the caller's handle from the
moment a job is submitted, and it must therefore exist before the two facts that make a run
recognisable do — which procedure adjudicates it and whose conduct it is about are both
resolved by the understanding stage, minutes later. So the label is a SECOND string, minted
once after understanding and never a rename: nothing keyed on the id moves, and a run with no
label reads as one that has not reached understanding yet rather than as a failure.

Every field is truncated or padded to its own width, so a label is exactly
:data:`LABEL_WIDTH` characters whatever it holds and a column of them aligns in a table, a
log line and a CSV alike. The padding is part of the value rather than the renderer's job:
the label travels through JSON, a job document and an export, and only one of those three has
a column to align in. Pad and separator are the same character, which costs nothing because
the fields are at fixed offsets and buys an unbroken run of them being legible as *this field
did not resolve* — the state a label must never fill in with a guess.
"""

import logging
import re
import secrets
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Minted-id alphabet. No I/L/O/U: an id is read off a screen and typed back, and a
#: transcription slip must not resolve to a different run.
_ID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

#: Length of a minted id. 32**10 keyspace, short enough to quote in a sentence.
ID_LENGTH = 10

#: Per-field widths, in label order: procedure, subject, event date, id tail.
PROCEDURE_WIDTH = 4
SUBJECT_WIDTH = 8
DATE_WIDTH = 6
TAIL_WIDTH = 4

#: Field filler and field separator, deliberately the same character (see the module docstring).
PAD = "-"
SEP = "-"

LABEL_WIDTH = PROCEDURE_WIDTH + SUBJECT_WIDTH + DATE_WIDTH + TAIL_WIDTH + 3 * len(SEP)

#: Digits then letters, for telling apart two procedures whose names share a prefix.
_DISAMBIGUATORS = "123456789ABCDEFGHJKMNPQRSTVWXYZ"

_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

#: One date, month or timestamp. Anchored and year-first, so an all-digit identifier is not
#: mistaken for one — an account number and a compacted date are the same characters.
_TEMPORAL_PART = re.compile(r"\d{4}[-/]\d{1,2}([-/]\d{1,2})?([T ][\d:.+Zz-]*)?$")

#: How a window writes its two ends. Read per part, because an ISO interval is a whole value
#: made of nothing but two timestamps.
_INTERVAL_SEP = re.compile(r"\s*(?:/|—|–|\bto\b)\s*")


def _is_temporal(value: str) -> bool:
    """Is ``value`` wholly a date, a timestamp or an interval of them?

    The label already carries a date field, so such a value names no party.
    """
    parts = [p for p in _INTERVAL_SEP.split(value) if p]
    return bool(parts) and all(_TEMPORAL_PART.fullmatch(p) for p in parts)


def new_incident_id() -> str:
    """A fresh incident id: :data:`ID_LENGTH` characters of :data:`_ID_ALPHABET`.

    Also a valid storage key segment, which is what an incident id has to be.
    """
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(ID_LENGTH))


def _fit(text: Any, width: int) -> str:
    """``text`` at exactly ``width``: upper-cased, non-alphanumerics dropped, then cut or padded."""
    clean = re.sub(r"[^A-Z0-9]", "", str(text or "").upper())
    return clean[:width].ljust(width, PAD)


def _attr(item: Any, name: str) -> str:
    """One field of an entity that may be a model or a dict (a replayed job document is dicts)."""
    if isinstance(item, dict):
        return str(item.get(name, "") or "")
    return str(getattr(item, name, "") or "")


def procedure_tokens(pack) -> Dict[str, str]:
    """``{ruleset key: PROCEDURE_WIDTH-char token}``, disambiguated within this pack.

    A truncation, so a reader can invert it by eye. Where two keys truncate the same way both
    get a numbered variant instead of one silently standing for the other — the label's job is
    to tell two runs apart, and a token shared by two procedures does the opposite.
    """
    keys: List[str] = []
    try:
        keys = [str(k) for k in (pack.ruleset_keys() if pack is not None else []) if str(k)]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read the pack's ruleset keys for labelling: %s", exc)
        return {}
    plain = {key: _fit(key, PROCEDURE_WIDTH) for key in keys}
    groups: Dict[str, List[str]] = {}
    for key, token in plain.items():
        groups.setdefault(token, []).append(key)
    out: Dict[str, str] = {}
    for token, members in groups.items():
        if len(members) == 1:
            out[members[0]] = token
            continue
        for index, key in enumerate(members):
            mark = _DISAMBIGUATORS[index % len(_DISAMBIGUATORS)]
            out[key] = token[: PROCEDURE_WIDTH - 1] + mark
    return out


def _procedure_field(pack, analysis) -> Tuple[str, str, str]:
    """``(token, detail, ruleset key)``; ``("", reason, "")`` where no procedure was selected.

    Resolved through the shared module-level selector, so the label names the ruleset the
    verdict will take rather than a second answer to the same question. A ``no_match`` leaves
    the field empty on purpose: the pack's default still adjudicates, but printing its token
    would claim a match no scorer made — the one thing a run's name must not do. The key is
    returned because the same ruleset also declares what its subject is.
    """
    if pack is None:
        return "", "no knowledge pack", ""
    if analysis is None:
        return "", "the incident was not understood", ""
    try:
        # Imported here rather than at module scope: `incident_input` reaches this module to
        # mint an id, and that path must not pull the correlation engine in with it.
        from correlation import select_correlation_spec_explained

        spec, basis = select_correlation_spec_explained(pack, analysis)
        if getattr(basis, "defaulted", False):
            return "", "no procedure matched this incident", ""
        use_case = str((spec or {}).get("use_case", "") or "")
        key = str(pack.ruleset_key_for(use_case) or "") if use_case else ""
        if not key:
            return "", "no procedure matched this incident", ""
        token = procedure_tokens(pack).get(key) or _fit(key, PROCEDURE_WIDTH)
        return token, "procedure=%s" % key, key
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not name the procedure for this run's label (%s).", exc)
        return "", "the procedure could not be resolved", ""


def _subject_field(pack, analysis, ruleset_key: str = "") -> Tuple[str, str]:
    """``(value, detail)`` for the entity this run is about; ``("", reason)`` when none resolved.

    The adjudicating ruleset's own ``subject_entity`` comes first, because that is the pack
    stating what its procedure is about — and it is the type the verdict engine anchors every
    subject on, so the label and the verdict name the same party. Then actor types in the
    pack's declaration order (the order attribution itself takes), then scope types, then
    whatever was extracted: a label names a person before a population.
    """
    entities = list(getattr(analysis, "extracted_entities", None) or [])
    if not entities:
        return "", "the incident named no entity"
    preferred: List[str] = []
    if pack is not None:
        # Two reads, two guards: a pack that cannot name its procedure's subject must still get
        # the actor-before-scope ordering, or one absent accessor silently costs both.
        if ruleset_key:
            try:
                declared = str((pack.ruleset_spec(ruleset_key) or {}).get("subject_entity") or "")
                preferred = [declared] if declared else []
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not read the procedure's subject for labelling: %s", exc)
        try:
            preferred += list(pack.actor_entity_types()) + list(pack.scope_entity_types())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read entity roles for labelling: %s", exc)
    skipped = ""
    for etype in preferred + [""]:
        for entity in entities:
            value = _attr(entity, "value").strip()
            if not value:
                continue
            found = _attr(entity, "type")
            if etype and found != etype:
                continue
            if _is_temporal(value):
                # A date is already the next field. Printing it twice names nobody, and a blank
                # subject an operator can read as "unidentified" is the more useful answer.
                skipped = skipped or "%s %s is a date, not a party" % (found or "?", value)
                continue
            # The local part of an address identifies; the domain is shared by everyone on it.
            return value.split("@", 1)[0], "subject=%s %s" % (found or "?", value)
    return "", skipped or "no extracted entity carried a value"


def _date_field(analysis, incident) -> Tuple[str, str]:
    """``(YYMMDD, detail)`` — the event window's start where the incident stated one, else its ingestion date."""
    window = getattr(analysis, "event_time", None)
    stated = str(getattr(window, "start", "") or "") if window is not None else ""
    for text, basis in ((stated, "event date"), (str((incident or {}).get("timestamp") or ""), "ingestion date")):
        match = _ISO_DATE.search(text)
        if match:
            year, month, day = match.groups()
            return year[2:] + month + day, "date=%s (%s)" % (match.group(0), basis)
    return "", "the incident carried no readable date"


def build_label(pack, analysis, incident) -> Tuple[str, str]:
    """``(label, detail)`` for one run: procedure, subject, event date and the id's own tail.

    The tail is what keeps two runs of the same procedure over the same subject on the same
    day apart, so the label is a handle and not just a description. ``detail`` explains each
    field, including the ones that did not resolve — a reader who cannot tell a truncated
    field from an absent one is being told something the run does not know.

    It is the id's LEADING characters, for two reasons that agree. Every other surface here
    abbreviates an id by its prefix (a job id prints as its first eight), so a tail read off a
    label is greppable against what an operator already sees; and a derived id carries its
    provenance as a SUFFIX, so the trailing characters are the one part that is shared —
    measured over the stored runs, every link child ended in the same four.
    """
    procedure, procedure_detail, ruleset_key = _procedure_field(pack, analysis)
    subject, subject_detail = _subject_field(pack, analysis, ruleset_key)
    date, date_detail = _date_field(analysis, incident)
    tail = re.sub(r"[^A-Z0-9]", "", str((incident or {}).get("id") or "").upper())[:TAIL_WIDTH]
    label = SEP.join(
        (
            _fit(procedure, PROCEDURE_WIDTH),
            _fit(subject, SUBJECT_WIDTH),
            _fit(date, DATE_WIDTH),
            _fit(tail, TAIL_WIDTH),
        )
    )
    return label, "; ".join((procedure_detail, subject_detail, date_detail))


def stamp_label(incident, pack, analysis) -> str:
    """Write ``label`` + ``label_detail`` onto ``incident`` and return the label.

    Called once per run, from the understanding stage's own call sites, so the job document,
    the exports and the report all carry the same string. Never raises and never overwrites a
    label already on the incident: a resumed or re-run stage would otherwise rename a run
    mid-flight, which is the failure this whole module exists to avoid.
    """
    if not isinstance(incident, dict):
        return ""
    existing = str(incident.get("label") or "")
    if existing:
        return existing
    try:
        label, detail = build_label(pack, analysis, incident)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build a label for this run (%s); its id stands alone.", exc)
        return ""
    incident["label"] = label
    incident["label_detail"] = detail
    logger.info("Run %s is labelled %s (%s)", incident.get("id"), label, detail)
    return label
