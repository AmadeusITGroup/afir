"""A namespaced view of a backend: every key is written and read with a prefix.

Listing strips the prefix back off — a caller gets out exactly what it put in.
"""

from pathlib import Path
from typing import Callable, List, Optional

from src.storage.base import StorageBackend, StoredObject, key_prefix, safe_key


class PrefixedStorage(StorageBackend):
    """Delegates to ``inner`` with every key under ``prefix``."""

    def __init__(self, inner: StorageBackend, prefix: str):
        self._inner = inner
        self._prefix = key_prefix(prefix)
        self.kind = f"{inner.kind}:{self._prefix}" if self._prefix else inner.kind

    @property
    def inner(self) -> StorageBackend:
        return self._inner

    @property
    def prefix(self) -> str:
        return self._prefix

    def _key(self, key: str) -> str:
        validated = safe_key(key)
        return f"{self._prefix}/{validated}" if self._prefix else validated

    # -- write -------------------------------------------------------------

    def put_text(
        self, key: str, text: str, verify: Optional[Callable[[str], object]] = None
    ) -> bool:
        return self._inner.put_text(self._key(key), text, verify=verify)

    def put_bytes(self, key: str, blob: bytes) -> bool:
        return self._inner.put_bytes(self._key(key), blob)

    def append_text(self, key: str, text: str) -> bool:
        return self._inner.append_text(self._key(key), text)

    # -- read --------------------------------------------------------------

    def get_text(self, key: str) -> Optional[str]:
        return self._inner.get_text(self._key(key))

    def get_bytes(self, key: str) -> Optional[bytes]:
        return self._inner.get_bytes(self._key(key))

    def get_previous_text(self, key: str) -> Optional[str]:
        return self._inner.get_previous_text(self._key(key))

    def exists(self, key: str) -> bool:
        return self._inner.exists(self._key(key))

    def list_keys(self, prefix: str) -> List[StoredObject]:
        scoped = key_prefix(prefix)
        full = "/".join(p for p in (self._prefix, scoped) if p)
        cut = len(self._prefix) + 1 if self._prefix else 0
        return [
            StoredObject(key=obj.key[cut:], size=obj.size, mtime=obj.mtime)
            for obj in self._inner.list_keys(full)
            if not cut or obj.key.startswith(f"{self._prefix}/")
        ]

    # -- housekeeping ------------------------------------------------------

    def delete(self, key: str) -> bool:
        return self._inner.delete(self._key(key))

    @property
    def degradation(self) -> Optional[str]:
        return self._inner.degradation

    def flush(self, timeout: float = 30.0) -> bool:
        """Forwarded to inner. Absent here, ``JobStore.flush`` would return ``True`` without
        waiting and void the gate-durability guarantee.
        """
        flush = getattr(self._inner, "flush", None)
        return True if flush is None else bool(flush(timeout=timeout))

    def close(self, timeout: float = 10.0) -> None:
        # Not forwarded: the wrapper does not own the backend; whoever built it closes it.
        return None

    @property
    def root(self):
        """Where this view maps to, when the inner backend can say. May be a ``Path`` or a string."""
        base = getattr(self._inner, "root", None)
        if base is None or not self._prefix:
            return base
        if isinstance(base, Path):
            return base / self._prefix
        return f"{str(base).rstrip('/')}/{self._prefix}"
