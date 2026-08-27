"""Retriever abstraction: ``DataRetriever.retrieve`` turns a ``RetrievalQuery`` into rows."""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List

from src.models.pydantic_models import RetrievalQuery

logger = logging.getLogger(__name__)

# Quoted `'<...>'` placeholder left in a generated query. Wildcards allowed around it so
# a LIKE template (`'<actor>%'`) is caught; `'%<script>%'` is ambiguous and also caught.
_PLACEHOLDER_LITERAL = re.compile(r"""(['"])[%_\s]*<[^'"<>]*>[%_\s]*\1""")


def unresolved_placeholders(text) -> List[str]:
    """Quoted ``'<...>'`` placeholder literals left in a generated query.

    Read by ``log_retrieval._gather`` for empty results only: a placeholder in a projection
    or ``ORDER BY`` is harmless, and rows that came back are real whatever the text looked like.
    """
    return [m.group(0) for m in _PLACEHOLDER_LITERAL.finditer(str(text or ""))]


def _key_enforced_for(retriever, text) -> bool:
    """True when ``text`` constrained every field of this source's resolved actor key.

    Never raises: every failure path returns False, since a wrong True fabricates a finding.
    """
    try:
        fields = list(getattr(retriever, "last_conjunction_fields", []) or [])
        if not fields:
            return False
        from src.retrievers.query_guards import (
            key_was_enforced,
            key_was_enforced_dsl,
        )

        stripped = str(text or "").lstrip()
        if stripped.startswith("{"):
            import json as _json

            try:
                return key_was_enforced_dsl(_json.loads(stripped), fields)
            except ValueError:
                return False
        return key_was_enforced(str(text or ""), fields)
    except Exception:  # noqa: BLE001 — a diagnostic must never break retrieval
        logger.debug(
            "key-enforcement check raised for %s",
            getattr(retriever, "config", {}).get("name", "?"),
            exc_info=True,
        )
        return False


def publish_query(retriever, text, on_query=None) -> None:
    """Record the final backend query and announce it via ``on_query``, pre-execution.

    Called after every guard so what is announced is what runs. A raising callback is
    swallowed. Also catches unresolved placeholders: the one seam every backend passes with
    its final text, so the check cannot be half-wired.
    """
    retriever.last_generated_query = text
    # Decides whether a zero-row result is a gap or a finding; read off the final text so
    # an unkeyed query does not claim to be one.
    retriever.last_key_enforced = _key_enforced_for(retriever, text)
    # Cleared here so a path that published without resolving does not inherit the previous key.
    retriever.last_conjunction_fields = []
    placeholders = unresolved_placeholders(text)
    if placeholders:
        logger.warning(
            "Source '%s': the generated query still contains unresolved placeholder(s) %s "
            "— it will run, match NOTHING, and report 0 rows as a SUCCESS. The value the "
            "predicate needed never reached the generation prompt; check that the query "
            "carries the entity (see field_mapping.render_identifiers) and that schema "
            "discovery for this source succeeded. Query: %s",
            getattr(retriever, "config", {}).get("name", "?"),
            ", ".join(placeholders),
            text,
        )
    if on_query is None:
        return
    try:
        on_query(text)
    except Exception:  # noqa: BLE001 — visibility must never break retrieval
        logger.debug(
            "on_query callback raised for %s",
            getattr(retriever, "config", {}).get("name", "?"),
            exc_info=True,
        )


class DataRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self, query: RetrievalQuery, guidance: str = "", on_query=None
    ) -> List[Dict]:
        """Fetch records for the given query; return a list of row dicts.

        ``guidance`` is per-call so one job's correction does not bleed into another's.
        ``on_query(text)`` is called with the final text before it is sent, after all guards;
        a raising callback must not fail the retrieval.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any connections/sessions held by the retriever."""
        ...
