"""Convert operator uploads for the pack assistant.

Documents become text before the model sees them; images stay bytes as content blocks.
Conversion routes through ``src/rag/document_ingester`` (one reader, no drift), each
upload spooled to ``exports_dir()/uploads`` — a Databricks App's filesystem is not
``/tmp``. Every cap is stated in the model's text, never applied silently.
"""

import base64
import binascii
import itertools
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.rag.document_ingester import DocumentIngester
from src.utils.paths import exports_dir

logger = logging.getLogger(__name__)

#: Suffixes already text — read directly, not through the ingester.
TEXT_SUFFIXES = {".md", ".markdown", ".yaml", ".yml", ".txt", ".text", ".log"}

#: Suffixes the ingester converts. Mirrored explicitly so an upstream change surfaces as a test failure.
DOCUMENT_SUFFIXES = {".pdf", ".docx", ".doc", ".csv", ".json"}

#: Image suffixes and media types. Explicit rather than ``mimetypes.guess_type``: the
#: media type goes into a data URL, so platform-dependent guesses are unsafe.
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

#: Characters of extracted text kept per attachment.
MAX_TEXT_CHARS = 40_000

#: Total across all attachments; beyond this the pack itself has no room.
MAX_TOTAL_TEXT_CHARS = 120_000

#: Per-upload byte cap. ``client_max_size`` is a higher ceiling; this is the policy.
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024

#: Lower than a document's: images are sent as base64 (~33% inflation) and billed as tokens.
MAX_IMAGE_BYTES = 4 * 1024 * 1024

#: Attachments per request.
MAX_ATTACHMENTS = 8


#: Deterministic counter so cleanup tests can inspect it.
_SPOOL_SEQ = itertools.count(1)


class AttachmentError(ValueError):
    """One upload could not be used. Carries a reason meant for an operator to read."""


def _safe_upload_name(name: str) -> str:
    """Sanitise a filename for use as a spool path (leaf only).

    Sanitised, not rejected: this names a scratch file, and refusing an upload over a
    bracket in its name would be an obstacle with no safety behind it.
    """
    leaf = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", leaf).lstrip(".")
    return (cleaned or "upload")[:120]


def classify(name: str) -> Tuple[str, str]:
    """``(kind, suffix)`` where kind is ``text``/``document``/``image``/``unsupported``."""
    suffix = ""
    leaf = _safe_upload_name(name)
    if "." in leaf:
        suffix = "." + leaf.rsplit(".", 1)[1].lower()
    if suffix in TEXT_SUFFIXES:
        return "text", suffix
    if suffix in DOCUMENT_SUFFIXES:
        return "document", suffix
    if suffix in IMAGE_TYPES:
        return "image", suffix
    return "unsupported", suffix


def _decode(raw: Any, *, name: str) -> bytes:
    """Base64-decode, accepting an optional ``data:<mime>;base64,`` prefix."""
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    text = str(raw or "")
    if not text:
        raise AttachmentError(f"{name}: no content")
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as e:
        raise AttachmentError(f"{name}: content is not valid base64 ({e})") from e


def _spool_dir():
    path = exports_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _extract_text(data: bytes, *, name: str, kind: str, suffix: str) -> str:
    """The attachment's text, by whichever of the two routes its suffix calls for."""
    if kind == "text":
        # Already text; decoded permissively so one bad byte doesn't refuse the whole upload.
        return data.decode("utf-8", errors="replace")

    # PID-prefixed to avoid collision; per-request subdir so readers can't echo the spool name.
    scratch = _spool_dir() / f"{os.getpid()}-{next(_SPOOL_SEQ)}"
    scratch.mkdir(parents=True, exist_ok=True)
    spool = scratch / _safe_upload_name(name)
    try:
        spool.write_bytes(data)
        doc = await DocumentIngester().ingest_file(str(spool))
        if not doc or not str(doc.get("content") or "").strip():
            # ingest_file returns None on any failure; empty result refused, not silently attached.
            raise AttachmentError(
                f"{name}: no text could be extracted (a scanned or protected "
                f"{suffix.lstrip('.') or 'file'} has no text layer — describe it in your "
                "question, or attach it as an image)"
            )
        return str(doc["content"])
    finally:
        # File then directory; best-effort: a leak doesn't fail an otherwise good upload.
        for path, remove in ((spool, os.unlink), (scratch, os.rmdir)):
            try:
                remove(path)
            except OSError:
                logger.warning("Could not remove the spooled upload at %s", path)


async def prepare(
    uploads: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``(attachments, errors)`` — convert every upload, refusing none on another's behalf.

    Not all-or-nothing: an attachment only adds context, so one bad file must not block
    the rest. Every rejection is returned so nothing is dropped quietly.
    """
    prepared: List[Dict[str, Any]] = []
    errors: List[str] = []
    if not uploads:
        return prepared, errors
    if len(uploads) > MAX_ATTACHMENTS:
        errors.append(
            f"{len(uploads)} files attached; only the first {MAX_ATTACHMENTS} were used"
        )
        uploads = uploads[:MAX_ATTACHMENTS]

    text_used = 0
    for upload in uploads:
        if not isinstance(upload, dict):
            errors.append("an attachment was not an object with a name and content")
            continue
        name = str(upload.get("name") or "attachment")
        kind, suffix = classify(name)
        if kind == "unsupported":
            errors.append(
                f"{name}: {suffix or 'that'} is not a readable format — "
                f"documents ({', '.join(sorted(DOCUMENT_SUFFIXES))}), text "
                f"({', '.join(sorted(TEXT_SUFFIXES))}) or images "
                f"({', '.join(sorted(IMAGE_TYPES))})"
            )
            continue
        try:
            data = _decode(upload.get("content"), name=name)
        except AttachmentError as e:
            errors.append(str(e))
            continue

        cap = MAX_IMAGE_BYTES if kind == "image" else MAX_ATTACHMENT_BYTES
        if len(data) > cap:
            errors.append(
                f"{name}: {len(data)} bytes is over the {cap}-byte limit for "
                f"{'an image' if kind == 'image' else 'a document'}"
            )
            continue
        if not data:
            errors.append(f"{name}: empty file")
            continue

        if kind == "image":
            prepared.append(
                {
                    "name": name,
                    "kind": "image",
                    "bytes": len(data),
                    "media_type": IMAGE_TYPES[suffix],
                    "data": data,
                    "note": "",
                }
            )
            continue

        try:
            text = await _extract_text(data, name=name, kind=kind, suffix=suffix)
        except AttachmentError as e:
            errors.append(str(e))
            continue
        except (
            Exception
        ) as e:  # noqa: BLE001 — a reader's own failure, reported not raised
            logger.warning("Attachment %s could not be read: %s", name, e)
            errors.append(f"{name}: could not be read ({e.__class__.__name__})")
            continue

        note = ""
        full_chars = len(text)
        if full_chars > MAX_TEXT_CHARS:
            # Capture original length before slicing; reporting len(text) would read as no truncation.
            text = text[:MAX_TEXT_CHARS]
            note = (
                f"truncated to the first {MAX_TEXT_CHARS} of {full_chars} "
                "characters — content past this point was NOT read"
            )
        remaining = MAX_TOTAL_TEXT_CHARS - text_used
        if remaining <= 0:
            errors.append(
                f"{name}: skipped, the {MAX_TOTAL_TEXT_CHARS}-character total "
                "attachment budget was already spent by earlier files"
            )
            continue
        if len(text) > remaining:
            text = text[:remaining]
            note = (
                f"cut to {remaining} characters — the total attachment budget "
                "ran out on this file"
            )
        text_used += len(text)
        prepared.append(
            {
                "name": name,
                "kind": kind,
                "bytes": len(data),
                "chars": len(text),
                "text": text,
                "note": note,
            }
        )
    return prepared, errors


def image_blocks(attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Image attachments as ``image_url`` content blocks with data URLs (no hosted address)."""
    blocks = []
    for att in attachments:
        if att.get("kind") != "image" or not att.get("data"):
            continue
        payload = base64.b64encode(att["data"]).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{att['media_type']};base64,{payload}"},
            }
        )
    return blocks


def image_placeholder(att: Dict[str, Any]) -> str:
    """Placeholder for an image the endpoint cannot read.

    Explicit rather than dropped so the assistant cannot answer as though it saw the diagram.
    """
    return (
        f"[image attached: {att.get('name')}, {att.get('bytes')} bytes, "
        f"{att.get('media_type')} — this endpoint cannot read images, so it was NOT "
        "seen. Describe what it shows in your question instead.]"
    )
