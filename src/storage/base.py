"""Named-blob contract for durable state. :func:`safe_key` raises rather than normalises.
Every method returns rather than raises (except key validation).
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

#: One key segment. Narrower than a filesystem allows; a non-matching value is a caller bug.
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")

#: Per-segment length cap. A key far past this comes only from unvalidated input.
MAX_SEGMENT_LEN = 160


@dataclass(frozen=True)
class StoredObject:
    """One entry from :meth:`StorageBackend.list_keys`.

    ``mtime`` and ``size`` are best-effort; ``0.0`` / ``0`` when unknown. Unknown does not mean absent.
    """

    key: str
    size: int = 0
    mtime: float = 0.0

    @property
    def name(self) -> str:
        """The last key segment; the filename portion for callers matching by name."""
        return self.key.rsplit("/", 1)[-1]


def safe_key(key: str) -> str:
    """Validate and normalise a blob key. Raises ``ValueError`` for absolute paths, ``..`` segments, backslashes, empty segments, or characters outside :data:`_SEGMENT`."""
    text = str(key or "").strip().replace("\\", "/")
    if not text or text.startswith("/"):
        raise ValueError(f"storage key must be relative and non-empty: {key!r}")
    segments = text.split("/")
    for segment in segments:
        if not segment or len(segment) > MAX_SEGMENT_LEN:
            raise ValueError(f"unusable segment in storage key {key!r}")
        if not _SEGMENT.fullmatch(segment):
            raise ValueError(f"unusable segment {segment!r} in storage key {key!r}")
    return "/".join(segments)


def key_prefix(prefix: str) -> str:
    """Validate a listing prefix (leading segments, no trailing slash). Empty means list everything."""
    text = str(prefix or "").strip()
    if not text:
        return ""
    # Trailing slash is cosmetic. A leading slash is not — stripping it would accept "/etc/passwd".
    text = text.rstrip("/")
    return safe_key(text) if text else ""


class StorageBackend:
    """The contract. See the module docstring; every implementation is best-effort."""

    #: Short identifier for logs and the health surface (``local`` / ``databricks``).
    kind = "abstract"

    # -- write -------------------------------------------------------------

    def put_text(
        self, key: str, text: str, verify: Optional[Callable[[str], object]] = None
    ) -> bool:
        """Store ``text`` at ``key``, replacing what was there. True when stored.

        ``verify`` runs on the read-back content, not the string in hand: a truncated
        write still valid as JSON must be caught at persist time.
        """
        raise NotImplementedError

    def put_bytes(self, key: str, blob: bytes) -> bool:
        """Store raw bytes at ``key``. True when stored. No ``verify``: artifacts have no cheap parse-back."""
        raise NotImplementedError

    def append_text(self, key: str, text: str) -> bool:
        """Append ``text`` to ``key``, creating it if absent. True when appended."""
        raise NotImplementedError

    # -- read --------------------------------------------------------------

    def get_text(self, key: str) -> Optional[str]:
        """The content at ``key``, or ``None`` when absent or unreadable (deliberately conflated)."""
        raise NotImplementedError

    def get_bytes(self, key: str) -> Optional[bytes]:
        """The raw bytes at ``key``, for a PDF or any other non-text artifact."""
        raise NotImplementedError

    def get_previous_text(self, key: str) -> Optional[str]:
        """The last content committed before the current one, or ``None``.

        On the interface because the policy is the caller's: ``JobStore.load_all`` falls
        back to this when the current document will not parse.
        """
        return None

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def list_keys(self, prefix: str) -> List[StoredObject]:
        """Everything directly or transitively under ``prefix``. Empty on failure; order unspecified."""
        raise NotImplementedError

    # -- housekeeping ------------------------------------------------------

    def delete(self, key: str) -> bool:
        """Remove ``key``. True when something was removed."""
        raise NotImplementedError

    # -- health ------------------------------------------------------------

    @property
    def degradation(self) -> Optional[str]:
        """Why durability is compromised, or ``None``. Surfaced rather than only logged: the operator cannot otherwise see this."""
        return None

    def close(self, timeout: float = 10.0) -> None:
        """Flush anything pending and release resources. Must not raise.

        ``timeout`` bounds the flush: a short window between ``SIGTERM`` and ``SIGKILL``.
        """
        return None
