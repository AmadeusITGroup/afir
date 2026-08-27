"""Analyst direction for one stage: renders gate-rejection corrections into a system message.

Ranked below the knowledge pack and any deterministic verdict, above distilled FeedbackLoop
guidance. Bounded so a repeatedly rejected stage cannot grow an unbounded prompt.
"""

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Newest items win; the latest correction is the one the analyst is currently applying.
MAX_ITEMS = 5
MAX_CHARS = 2000
_MAX_ITEM_CHARS = 600

_PREAMBLE = (
    "ANALYST DIRECTION FOR THIS STAGE (a human reviewed this stage's output on this "
    "incident and rejected it, with the corrections below). Apply them. They outrank "
    "general learned guidance. They do NOT outrank the knowledge pack, the official "
    "procedure, or a deterministic verdict, and they do not license a conclusion the "
    "retrieved data does not support — if a correction cannot be satisfied by the "
    "data, say so plainly instead of asserting it."
)


def render_guidance(items: Optional[Sequence[str]]) -> str:
    """Analyst corrections as a system-message body, or ``""`` when none. Never raises."""
    try:
        if not isinstance(items, (list, tuple)):
            return ""
        cleaned = []
        for item in items:
            text = str(item or "").strip()
            if text:
                cleaned.append(text[:_MAX_ITEM_CHARS])
        if not cleaned:
            return ""
        kept = cleaned[-MAX_ITEMS:]
        lines = [_PREAMBLE, ""]
        lines += [f"- {text}" for text in kept]
        rendered = "\n".join(lines)
        if len(rendered) > MAX_CHARS:
            rendered = rendered[:MAX_CHARS].rstrip() + " […]"
        return rendered
    except Exception as exc:  # noqa: BLE001 — advisory, never fatal
        logger.warning("Could not render analyst guidance (%s); ignoring it.", exc)
        return ""


def guidance_message(items: Optional[Sequence[str]]) -> Optional[dict]:
    """The rendered guidance as a chat system message, or ``None``."""
    body = render_guidance(items)
    return {"role": "system", "content": body} if body else None


def guidance_prompt_line(guidance: Optional[str]) -> str:
    """Direction inline in a concatenated system string; pack guards are enforced after generation."""
    try:
        text = str(guidance or "").strip()
        if not text:
            return ""
        return (
            "ANALYST DIRECTION for this query (a human rejected the previous attempt; "
            "apply it unless it conflicts with the mandatory rules above): "
            + text[:MAX_CHARS]
            + "\n"
        )
    except Exception as exc:  # noqa: BLE001 — advisory, never fatal
        logger.warning("Could not render analyst query guidance (%s); ignoring.", exc)
        return ""
