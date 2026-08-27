"""Local-disk storage: the default backend and the behavioural oracle for every other one.

Write path: temp file, ``fsync``, parse-back, ``.prev`` copy, then ``os.replace`` over the
target. The layout is the one that shipped before the seam existed: ``<root>/jobs/<id>.json``,
``<root>/exports/fraud_report_<id>.md``, and so on. Each step is load-bearing:

- Parse-back before committing. A truncated write that is still valid JSON is
  indistinguishable from a real document at load time.
- ``.prev`` is a copy, not a rename, so the target stays present throughout and a crash
  mid-write leaves a readable current document rather than only a backup. ``job_store``'s
  ``load_all`` falls back to it for exactly this reason.
- ``os.replace`` is atomic on POSIX, so a concurrent reader sees the old document or the new
  one and never a partial write. It does not exist on a Unity Catalog Volume, which is why
  the remote backend is a separate implementation.

Neither a ``.prev`` nor a ``.tmp`` file is a stored object: :meth:`list_keys` hides both, or a
caller enumerating jobs finds two entries for one job.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional

from src.storage.base import StorageBackend, StoredObject, key_prefix, safe_key
from src.utils.paths import data_dir

logger = logging.getLogger(__name__)

#: Suffixes this backend manages itself and never reports as stored objects.
_INTERNAL_SUFFIXES = (".prev",)
_TMP_MARKER = ".tmp"


class LocalStorage(StorageBackend):
    """Named blobs as files under one root. Never raises except on a bad key."""

    kind = "local"

    def __init__(self, root=None):
        # Defaults to data_dir() so the existing AFIR_DATA_DIR override keeps working
        # and an upgraded VM finds its own files where it left them.
        self._root = Path(root) if root else None

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else data_dir()

    def path_for(self, key: str) -> Path:
        """The absolute path a key resolves to. Public so tests can assert the layout."""
        return self.root / safe_key(key)

    # -- write -------------------------------------------------------------

    def put_text(
        self, key: str, text: str, verify: Optional[Callable[[str], object]] = None
    ) -> bool:
        path = self.path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Cannot create %s: %s", path.parent, exc)
            return False

        tmp = path.with_suffix(path.suffix + f"{_TMP_MARKER}{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            if verify is not None:
                # Read back from disk, not from `text`: see the module docstring.
                with open(tmp, "r", encoding="utf-8") as fh:
                    verify(fh.read())
        except Exception as exc:  # noqa: BLE001 — never fail a run over storage
            logger.warning("Failed writing %s: %s", key, exc)
            _unlink_quietly(tmp)
            return False

        try:
            if path.exists():
                shutil.copy2(path, path.with_suffix(path.suffix + ".prev"))
        except OSError as exc:
            logger.debug("Could not refresh .prev for %s: %s", key, exc)

        try:
            os.replace(tmp, path)
            return True
        except OSError as exc:
            logger.warning("Failed to swap in %s: %s", key, exc)
            _unlink_quietly(tmp)
            return False

    def append_text(self, key: str, text: str) -> bool:
        path = self.path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)
            return True
        except OSError as exc:
            logger.error("Failed appending to %s: %s", key, exc)
            return False

    # -- read --------------------------------------------------------------

    def get_text(self, key: str) -> Optional[str]:
        path = self.path_for(key)
        try:
            if not path.is_file():
                return None
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", key, exc)
            return None

    def get_bytes(self, key: str) -> Optional[bytes]:
        path = self.path_for(key)
        try:
            if not path.is_file():
                return None
            return path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", key, exc)
            return None

    def get_previous_text(self, key: str) -> Optional[str]:
        path = self.path_for(key)
        try:
            prev = path.with_suffix(path.suffix + ".prev")
            return prev.read_text(encoding="utf-8") if prev.is_file() else None
        except OSError as exc:
            logger.warning("Cannot read the previous version of %s: %s", key, exc)
            return None

    def exists(self, key: str) -> bool:
        try:
            return self.path_for(key).is_file()
        except OSError:
            return False

    def list_keys(self, prefix: str) -> List[StoredObject]:
        validated = key_prefix(prefix)
        base = (self.root / validated) if validated else self.root
        out: List[StoredObject] = []
        try:
            if not base.is_dir():
                return []
            entries = sorted(base.rglob("*"))
        except OSError as exc:
            logger.warning("Cannot list %s: %s", prefix, exc)
            return []
        for entry in entries:
            try:
                if not entry.is_file() or _is_internal(entry.name):
                    continue
                stat = entry.stat()
                rel = entry.relative_to(self.root).as_posix()
                out.append(
                    StoredObject(key=rel, size=stat.st_size, mtime=stat.st_mtime)
                )
            except OSError as exc:
                logger.debug("Skipping %s while listing: %s", entry, exc)
        return out

    # -- housekeeping ------------------------------------------------------

    def delete(self, key: str) -> bool:
        path = self.path_for(key)
        removed = _unlink_quietly(path)
        # The .prev is this backend's own artifact, so deleting a key removes it too;
        # otherwise load_all's .prev fallback would resurrect a deleted job.
        _unlink_quietly(path.with_suffix(path.suffix + ".prev"))
        return removed

    # -- binary write ------------------------------------------------------

    def put_bytes(self, key: str, blob: bytes) -> bool:
        """Store raw bytes without parse-back; a PDF has no cheap validity check."""
        path = self.path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + f"{_TMP_MARKER}{os.getpid()}")
            with open(tmp, "wb") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        except OSError as exc:
            logger.warning("Failed writing %s: %s", key, exc)
            return False


def _is_internal(name: str) -> bool:
    return name.endswith(_INTERNAL_SUFFIXES) or _TMP_MARKER in name


def _unlink_quietly(path) -> bool:
    try:
        if path is not None and path.exists():
            path.unlink()
            return True
    except OSError as exc:
        logger.debug("Could not remove %s: %s", getattr(path, "name", path), exc)
    return False


def json_verifier(text: str):
    """The ``verify`` callable for a JSON document. Raises on anything unparseable."""
    return json.loads(text)
