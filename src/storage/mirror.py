"""Config and knowledge-pack durability: mirror the trees, do not move them.

Local filesystem stays the working copy; this module adds sync down at boot and sync up
after each write. The store holds EDITS only — a delete is an edit too (see
:data:`TOMBSTONE_KEY`). Push failure is reported via ``durable`` and
:attr:`TreeMirror.degradation`, never raised. ``None`` for a local backend.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from src.storage.base import StorageBackend
from src.storage.prefixed import PrefixedStorage
from src.utils.paths import REPO_ROOT, config_dir

logger = logging.getLogger(__name__)

#: Where each tree's blobs live under the backend. Both are valid key segments.
CONFIG_PREFIX = "config"
KNOWLEDGE_PREFIX = "knowledge"

#: Deleted-paths record at the root of each prefix; named so it cannot be mistaken for content.
TOMBSTONE_KEY = "mirror_deleted.json"

#: Load-bearing hidden dirs and their key aliases (key segments must start with a letter or digit).
_HIDDEN_ALIASES = {".history": "history"}
_ALIAS_BACK = {v: k for k, v in _HIDDEN_ALIASES.items()}

#: Backend mechanics never pushed; ``.bak`` in particular holds pre-redaction credentials.
_SKIP_NAME = re.compile(r"(\.tmp\d+|\.bak|\.prev|~)$")

#: Files larger than this are not mirrored (uploading them on every save would stall the editor).
MAX_FILE_BYTES = 8_388_608

#: Config tree suffixes. Depth-1 keeps ``config/templates/`` out (reference material, never edited).
_CONFIG_SUFFIXES = frozenset({".yaml", ".yml"})

#: Shipped defaults; the seeding fallback of last resort in a deployed bundle.
_CONFIG_TEMPLATE_DIR = "templates"


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


class TreeMirror:
    """One local directory tree, mirrored to one prefix of a storage backend."""

    def __init__(
        self,
        storage: StorageBackend,
        prefix: str,
        local_root,
        *,
        label: str = "",
        suffixes: Optional[frozenset] = None,
        max_depth: int = 6,
    ):
        self._store = PrefixedStorage(storage, prefix)
        self._root = Path(local_root)
        self.label = label or prefix
        self._suffixes = suffixes
        self._max_depth = int(max_depth)
        self._failures: List[str] = []
        self._tombstones: set = set()
        self._tombstones_loaded = False

    # -- identity ----------------------------------------------------------

    @property
    def local_root(self) -> Path:
        return self._root

    @property
    def store(self) -> PrefixedStorage:
        return self._store

    def owns(self, path) -> bool:
        """Whether ``path`` lies inside this mirror's tree. Both sides resolved to handle symlinks."""
        try:
            Path(path).resolve().relative_to(self._root.resolve())
            return True
        except (ValueError, OSError):
            return False

    def rel_of(self, path) -> Optional[PurePosixPath]:
        try:
            return PurePosixPath(Path(path).resolve().relative_to(self._root.resolve()))
        except (ValueError, OSError):
            return None

    # -- inclusion ---------------------------------------------------------

    def includes(self, rel: PurePosixPath) -> bool:
        """Whether ``rel`` is a file this mirror carries. Logs nothing; a filter."""
        parts = rel.parts
        if not parts or len(parts) > self._max_depth:
            return False
        for i, part in enumerate(parts):
            if _SKIP_NAME.search(part):
                return False
            if _is_hidden(part) and part not in _HIDDEN_ALIASES:
                return False
            if part in _ALIAS_BACK and _ALIAS_BACK[part] not in parts[:i]:
                # A real dir named like the alias: refuse rather than collide in the undo namespace.
                return False
        if self._suffixes is not None and rel.suffix.lower() not in self._suffixes:
            # Pack tree has no suffix filter: history blobs have no extension.
            return False
        return True

    def _key_for(self, rel: PurePosixPath) -> Optional[str]:
        segments = [_HIDDEN_ALIASES.get(p, p) for p in rel.parts]
        return "/".join(segments) if segments else None

    def _rel_for(self, key: str) -> Optional[PurePosixPath]:
        parts = [p for p in str(key).split("/") if p]
        if not parts:
            return None
        return PurePosixPath(*[_ALIAS_BACK.get(p, p) for p in parts])

    # -- tombstones --------------------------------------------------------

    def _load_tombstones(self) -> set:
        if not self._tombstones_loaded:
            self._tombstones_loaded = True
            raw = self._store.get_text(TOMBSTONE_KEY)
            if raw:
                try:
                    doc = json.loads(raw)
                    paths = doc.get("deleted") if isinstance(doc, dict) else None
                    self._tombstones = {str(p) for p in (paths or []) if p}
                except Exception as exc:  # noqa: BLE001 - a corrupt record is not fatal
                    logger.warning(
                        "Unreadable %s tombstone record (%s); treating it as empty, so a "
                        "deleted file may come back from the bundle.",
                        self.label,
                        exc,
                    )
        return self._tombstones

    def _save_tombstones(self) -> bool:
        payload = json.dumps(
            {"deleted": sorted(self._tombstones)}, indent=2, sort_keys=True
        )
        return bool(self._store.put_text(TOMBSTONE_KEY, payload, verify=json.loads))

    # -- down --------------------------------------------------------------

    def sync_down(self) -> Dict[str, int]:
        """Lay the durable store's contents over the working copy. Returns counts for the boot log."""
        report = {"downloaded": 0, "deleted": 0, "skipped": 0, "failed": 0}
        for obj in self._store.list_keys(""):
            if obj.key == TOMBSTONE_KEY:
                continue
            rel = self._rel_for(obj.key)
            if rel is None or not self.includes(rel):
                report["skipped"] += 1
                continue
            blob = self._store.get_bytes(obj.key)
            if blob is None:
                logger.warning("%s: %s is listed but unreadable", self.label, obj.key)
                report["failed"] += 1
                continue
            target = self._root / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
                report["downloaded"] += 1
            except OSError as exc:
                logger.error("%s: cannot write %s: %s", self.label, target, exc)
                report["failed"] += 1

        for rel_text in sorted(self._load_tombstones()):
            rel = PurePosixPath(rel_text)
            if not self.includes(rel):
                continue
            target = self._root / rel
            try:
                if target.is_file():
                    target.unlink()
                    report["deleted"] += 1
            except OSError as exc:
                logger.warning("%s: cannot remove %s: %s", self.label, target, exc)
                report["failed"] += 1
        return report

    # -- up ----------------------------------------------------------------

    def push_path(self, path) -> bool:
        """Push one file by absolute path. Never raises.
        ``True`` also for paths this mirror is not responsible for.
        """
        rel = self.rel_of(path)
        if rel is None:
            return True
        if not self.includes(rel):
            logger.debug("%s: not mirroring %s", self.label, rel)
            return True
        key = self._key_for(rel)
        source = Path(path)
        try:
            size = source.stat().st_size
        except OSError as exc:
            return self._fail(f"{rel} vanished before it could be mirrored: {exc}")
        if size > MAX_FILE_BYTES:
            return self._fail(
                f"{rel} is {size} bytes, over the {MAX_FILE_BYTES}-byte mirror cap; "
                "it stays on the container's disk and will not survive a restart"
            )
        try:
            blob = source.read_bytes()
        except OSError as exc:
            return self._fail(f"{rel} could not be read back for mirroring: {exc}")
        if not self._store.put_bytes(key, blob):
            return self._fail(f"{rel} could not be stored durably")
        if str(rel) in self._load_tombstones():
            # Recreated: clear the tombstone so the next boot does not delete it again.
            self._tombstones.discard(str(rel))
            self._save_tombstones()
        self._clear_failures()
        return True

    def delete_path(self, path) -> bool:
        """Record a delete: remove the blob and remember the removal. Never raises."""
        rel = self.rel_of(path)
        if rel is None:
            return True
        if not self.includes(rel):
            return True
        key = self._key_for(rel)
        self._store.delete(key)
        self._load_tombstones().add(str(rel))
        if not self._save_tombstones():
            return self._fail(
                f"{rel} was deleted locally but the deletion could not be recorded; it "
                "will come back from the bundle after a restart"
            )
        self._clear_failures()
        return True

    def push_tree(self) -> Dict[str, int]:
        """Push every carried file. Not used at boot; for initial migration onto a durable store."""
        report = {"pushed": 0, "failed": 0}
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            rel = self.rel_of(path)
            if rel is None or not self.includes(rel):
                continue
            if self.push_path(path):
                report["pushed"] += 1
            else:
                report["failed"] += 1
        return report

    # -- health ------------------------------------------------------------

    def _fail(self, message: str) -> bool:
        logger.error("%s mirror: %s", self.label, message)
        self._failures.append(message)
        del self._failures[:-5]
        return False

    def _clear_failures(self) -> None:
        self._failures = []

    @property
    def degradation(self) -> Optional[str]:
        """Why this tree is not durable, or ``None``. Includes the backend's own degradation."""
        parts = []
        inner = self._store.degradation
        if inner:
            parts.append(inner)
        if self._failures:
            parts.append(
                f"{self.label}: " + "; ".join(self._failures[-3:])  # newest few
            )
        return " | ".join(parts) if parts else None


class MirrorSet:
    """The config tree and the pack tree, mirrored together through the same two hooks."""

    def __init__(self, config_mirror: TreeMirror, knowledge_mirror: TreeMirror):
        self.config = config_mirror
        self.knowledge = knowledge_mirror

    @property
    def mirrors(self) -> Tuple[TreeMirror, TreeMirror]:
        return (self.config, self.knowledge)

    def sync_down(self) -> Dict[str, Dict[str, int]]:
        return {m.label: m.sync_down() for m in self.mirrors}

    def install(self) -> None:
        """Wire the two write paths to this mirror. Imports deferred to avoid heavy boot-time deps."""
        from src import config_store
        from src.knowledge import pack_store

        config_store.set_mirror(self.config)
        pack_store.set_mirror(self.knowledge)

    @property
    def degradation(self) -> Optional[str]:
        parts = [m.degradation for m in self.mirrors]
        joined = [p for p in parts if p]
        return " | ".join(joined) if joined else None


# ------------------------------------------------------------------------ seeding


def seed_working_copies(
    *, config_working=None, knowledge_working=None
) -> Dict[str, int]:
    """Seed the writable working copies from the bundle. Copies only what is missing.

    No-op locally (working root IS bundle root). In a deployed App, ``config/templates/``
    is the fallback source: live ``config/*.yaml`` are gitignored and absent from the bundle.
    """
    from src.knowledge.pack_store import packs_root

    report = {"config": 0, "knowledge": 0}
    pairs = (
        (
            "config",
            REPO_ROOT / "config",
            Path(config_working) if config_working else config_dir(),
            _CONFIG_SUFFIXES,
            1,
        ),
        (
            "knowledge",
            REPO_ROOT / "knowledge",
            Path(knowledge_working) if knowledge_working else packs_root(),
            None,
            6,
        ),
    )
    for label, bundle, working, suffixes, depth in pairs:
        try:
            same = bundle.resolve() == working.resolve()
        except OSError:
            same = False
        if same or not bundle.is_dir():
            continue
        report[label] = _copy_missing(bundle, working, suffixes, depth, label)
        if label == "config":
            # Templates supply what depth 1 missed, flattened (``templates/main_config.yaml`` → ``main_config.yaml``).
            report[label] += _copy_missing(
                bundle / _CONFIG_TEMPLATE_DIR, working, suffixes, 1, "config template"
            )
        logger.info(
            "Seeded %d %s file(s) from the bundle at %s into the working copy at %s",
            report[label],
            label,
            bundle,
            working,
        )
    return report


def _copy_missing(bundle: Path, working: Path, suffixes, depth: int, label: str) -> int:
    copied = 0
    if not bundle.is_dir():
        # A tree that is not there seeds nothing. Reached for the template directory,
        # which an operator may legitimately have deleted from their own checkout.
        return copied
    for source in sorted(bundle.rglob("*")):
        if not source.is_file():
            continue
        rel = PurePosixPath(source.relative_to(bundle))
        if len(rel.parts) > depth:
            continue
        if any(_is_hidden(p) or _SKIP_NAME.search(p) for p in rel.parts):
            continue
        if suffixes is not None and rel.suffix.lower() not in suffixes:
            continue
        target = working / rel
        if target.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        except OSError as exc:
            logger.error("Could not seed %s %s: %s", label, rel, exc)
    return copied


# ------------------------------------------------------------------------ building


def build_mirrors(
    config: Optional[dict], storage: StorageBackend
) -> Optional[MirrorSet]:
    """The mirror set for this deployment, or ``None`` (local backend, or ``never`` override).

    ``storage.mirror_config_and_pack`` accepts ``auto`` / ``always`` / ``never``; auto mirrors
    only when the backend is remote.
    """
    from src.knowledge.pack_store import packs_root

    cfg = ((config or {}).get("storage") or {}) if isinstance(config, dict) else {}
    mode = str(cfg.get("mirror_config_and_pack") or "auto").strip().lower()
    remote = getattr(storage, "kind", "local") != "local"
    if mode in ("never", "off", "false", "no"):
        return None
    if mode not in ("always", "on", "true", "yes") and not remote:
        return None
    if mode not in ("auto", "always", "on", "true", "yes"):
        logger.error(
            "Unknown storage.mirror_config_and_pack %r; treating it as 'auto'. Valid "
            "values are 'auto', 'always' and 'never'.",
            mode,
        )
        if not remote:
            return None
    return MirrorSet(
        TreeMirror(
            storage,
            CONFIG_PREFIX,
            config_dir(),
            label="config",
            suffixes=_CONFIG_SUFFIXES,
            max_depth=1,
        ),
        TreeMirror(
            storage,
            KNOWLEDGE_PREFIX,
            packs_root(),
            label="knowledge",
            suffixes=None,
            max_depth=6,
        ),
    )


def working_copies_are_writable() -> Dict[str, bool]:
    """Whether each working root can be written to. Probes the real directory, not a stand-in."""
    from src.knowledge.pack_store import packs_root

    out = {}
    for label, root in (("config", config_dir()), ("knowledge", packs_root())):
        probe = Path(root) / f".afir_write_probe_{os.getpid()}"
        try:
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
            out[label] = True
        except OSError:
            out[label] = False
    return out
