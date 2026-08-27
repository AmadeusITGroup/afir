"""Pluggable knowledge-source abstraction.

A ``KnowledgeSource`` is anything that can yield documents for the unified knowledge base: the
domain playbooks, a local document folder, a Confluence space, a Unity Catalog table. Every
source emits the same document schema, so once consolidated the LLM sees one uniform knowledge
base whatever each document's origin. The dict every ``fetch()`` returns:

    title    : str  human-readable label
    content  : str  the text embedded and shown to the LLM
    type     : str  semantic tag, e.g. "playbook", "confluence_page"
    metadata : dict source-specific extras, no required keys

``KnowledgeBaseManager.add_documents`` stamps ``source``/``added_at``/``id`` on top, so a
source must not set those keys.

To add a backend: subclass ``KnowledgeSource``, implement ``source_name`` and ``fetch()``, and
register a ``type:`` for it in ``main.py``'s source factory.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

# One document, in the homogeneous schema every source must produce.
KnowledgeDoc = Dict[str, Any]


class KnowledgeSource(ABC):
    """Base class for every pluggable knowledge source."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Stable identifier used as the ``KnowledgeBaseManager`` source key.

        Must be unique across enabled sources — it is the key for idempotent
        per-source re-ingest (remove-then-add) and for fallback lookup.
        """
        ...

    @property
    def enabled(self) -> bool:
        """Whether this source should be ingested. Override to gate on config."""
        return True

    @abstractmethod
    async def fetch(self) -> List[KnowledgeDoc]:
        """Return all documents this source currently holds.

        Must be idempotent (no side effects; same set on repeat calls). Raise on
        an unrecoverable error so the orchestrator can log+skip this one source;
        return ``[]`` for an empty-but-healthy source.
        """
        ...
