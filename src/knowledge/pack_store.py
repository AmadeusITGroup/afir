"""File CRUD over a knowledge pack, with writes that cannot silently empty it.

Every write: re-parses off disk, checks frontmatter, and refuses if the result is empty
when the pre-image was not. Paths rejected, never sanitised. Mutating functions report
``durable`` and push the snapshot before the file it belongs to.
"""

import difflib
import hashlib
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

import yaml

from src.utils.paths import REPO_ROOT, knowledge_pack_dir

logger = logging.getLogger(__name__)

#: Set by ``MirrorSet.install()`` in App mode; ``None`` on every local deployment.
#: Module-level because there is one pack root per process.
_MIRROR = None


def set_mirror(mirror) -> None:
    """Install (or clear, with ``None``) the durable mirror for the pack tree."""
    global _MIRROR
    _MIRROR = mirror


def mirror() -> object:
    """The installed mirror, or ``None``. For the health surface and the tests."""
    return _MIRROR


def _mirror_push(path: Path) -> bool:
    """Push one written file to durable storage. ``True`` when nothing was at risk."""
    return True if _MIRROR is None else bool(_MIRROR.push_path(path))


def _mirror_delete(path: Path) -> bool:
    """Record a delete durably. ``True`` when nothing was at risk."""
    return True if _MIRROR is None else bool(_MIRROR.delete_path(path))


#: Suffixes the editor shows as text and accepts writes for. ``.txt`` is for
#: free-form notes an author may leave beside a ruleset.
EDITABLE_SUFFIXES = frozenset({".yaml", ".yml", ".md", ".txt"})

#: Per-pack snapshot store. Gitignored, and refused as a path segment so the editor
#: cannot be pointed at its own undo history.
HISTORY_DIR = ".history"

#: Above this size a file is download-only. Generated schema inventories exceed this
#: and must not be round-tripped through a browser control.
INLINE_EDIT_MAX_BYTES = 262_144

#: Hard caps. ``client_max_size`` is a higher ceiling; every handler states its own.
READ_MAX_BYTES = 1_048_576
WRITE_MAX_BYTES = 2_097_152

#: 6 covers the deepest real structure while refusing paths that wander.
MAX_DEPTH = 6

#: One path segment: letters, digits, ``.``, ``_``, ``-`` only.
_SEGMENT = re.compile(r"[A-Za-z0-9._\-]{1,128}")

#: A vocabulary word may contain a space but nothing that needs quoting in YAML.
_VOCAB_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._\-]{0,63}")

#: Temp files left mid-write; excluded from the tree so a crash doesn't present a partial write.
_TMP_SUFFIX = re.compile(r"\.tmp\d+$")

#: Where the pack template lives. Read by :func:`scaffold_pack`.
TEMPLATE_DIR = REPO_ROOT / "docs" / "knowledge-pack-template"

#: Template files a new pack skips: ``README.md`` describes the template itself;
#: ``domain_vocabulary.yaml`` must come from the caller's words, not the template's.
_TEMPLATE_SKIP = frozenset({"README.md", "domain_vocabulary.yaml"})


# --------------------------------------------------------------------------- errors


class PackStoreError(ValueError):
    """Base for every refusal here. A ``ValueError`` so an HTTP layer maps it to 400."""


class PackNotFound(PackStoreError):
    pass


class PackFileNotFound(PackStoreError):
    pass


class PackFileExists(PackStoreError):
    pass


class PackConflict(PackStoreError):
    """The file on disk is not the one the caller computed their edit against."""


class PackTooLarge(PackStoreError):
    """Over a declared cap. Distinct so the HTTP layer can answer 413 rather than 400."""


class PackWriteRejected(PackStoreError):
    """The candidate content failed verification. Carries the parser's own message."""

    def __init__(self, message: str, *, path: str = "", line: int = 0):
        super().__init__(message)
        self.path = path
        self.line = line


# ------------------------------------------------------------------- path handling


def packs_root() -> Path:
    """Parent of every installed pack. Derived from :func:`knowledge_pack_dir` so an
    ``AFIR_KNOWLEDGE_DIR`` override moves both the loader and the editor together."""
    return knowledge_pack_dir("_").parent


def safe_pack_name(name: str) -> str:
    """Validate a pack directory name, or raise.

    Leading dot refused separately: ``.history`` matches the segment pattern and would
    let the editor address its own undo store.
    """
    text = str(name or "").strip()
    if not text or not _SEGMENT.fullmatch(text) or ".." in text:
        raise PackStoreError(f"invalid pack name {name!r}")
    if text.startswith("."):
        raise PackStoreError(f"invalid pack name {name!r}: may not start with '.'")
    return text


def safe_rel_path(rel: str) -> PurePosixPath:
    """Validate a pack-relative file path, or raise.

    Rejects, never repairs. Tests ``".."`` as a full segment, not a substring — a
    real filename may contain ``..`` as characters.
    """
    text = str(rel or "").strip()
    if not text:
        raise PackStoreError("a file path is required")
    if text.startswith("/") or text.startswith("\\") or ":" in text:
        # Absolute path: refuse rather than strip the root and route to an unintended file.
        raise PackStoreError(f"invalid path {rel!r}: must be relative to the pack")
    if "\\" in text:
        # Backslash is a separator on some platforms and a legal character on others.
        raise PackStoreError(f"invalid path {rel!r}: use '/' as the separator")
    segments = text.split("/")
    if len(segments) > MAX_DEPTH:
        raise PackStoreError(f"invalid path {rel!r}: more than {MAX_DEPTH} levels deep")
    for seg in segments:
        if not _SEGMENT.fullmatch(seg):
            raise PackStoreError(f"invalid path {rel!r}: bad segment {seg!r}")
        if seg in (".", ".."):
            raise PackStoreError(f"invalid path {rel!r}: '{seg}' is not a file name")
    if segments[0] == HISTORY_DIR:
        raise PackStoreError(f"invalid path {rel!r}: {HISTORY_DIR}/ is not editable")
    return PurePosixPath(text)


def pack_dir(pack: str) -> Path:
    return knowledge_pack_dir(safe_pack_name(pack))


def pack_file(pack: str, rel: str) -> Path:
    """Absolute path of one file inside a pack. Validates both halves."""
    return pack_dir(pack) / safe_rel_path(rel)


def require_editable_suffix(rel: PurePosixPath) -> None:
    if rel.suffix.lower() not in EDITABLE_SUFFIXES:
        raise PackStoreError(
            f"{rel}: only {', '.join(sorted(EDITABLE_SUFFIXES))} files are editable"
        )


def classify(rel: str) -> str:
    """Classification for the UI help panel, structural where possible.

    ``other`` is for attachments ``pack_tree`` already marks non-editable.
    """
    p = PurePosixPath(str(rel))
    parts = p.parts
    name = p.name
    if "shared" in parts and "checks" in parts:
        return "shared_check"
    if "shared" in parts and "concepts" in parts:
        return "concept"
    if "playbooks" in parts:
        return "playbook"
    if "concepts" in parts:
        return "concept"
    if "cases" in parts:
        return "case"
    if "schemas" in parts:
        return "schema"
    if "data" in parts:
        return "data"
    if name == "entity_glossary.yaml":
        return "glossary"
    if name == "source_catalog.yaml":
        return "catalog"
    if name in ("rules.yaml", "rulesets.yaml"):
        return "ruleset"
    if name == "reporting.yaml":
        return "reporting"
    if name == "domain_vocabulary.yaml":
        return "vocabulary"
    suffix = p.suffix.lower()
    if suffix in (".md", ".txt"):
        return "notes"
    if suffix in (".yaml", ".yml"):
        return "reference"
    return "other"


# --------------------------------------------------------------------- verification


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _yaml_error_line(exc: Exception) -> int:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line = getattr(mark, "line", None)
    return int(line) + 1 if isinstance(line, int) else 0


def _is_empty_document(loaded: Any) -> bool:
    """True when the loader sees this as absent — ``None``, ``{}``, or ``[]``."""
    return loaded is None or loaded == {} or loaded == []


def _parse_yaml_or_raise(text: str, *, path: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PackWriteRejected(
            f"{path}: the content is not valid YAML: {exc}",
            path=path,
            line=_yaml_error_line(exc),
        ) from exc


def _parse_frontmatter_or_raise(text: str, *, path: str) -> None:
    """A leading ``---`` fenced block must parse. Mirrors ``pack._parse_frontmatter``
    so the loader's own reading of the boundary is what gets checked."""
    if not text.startswith("---"):
        return
    end = text.find("\n---", 3)
    if end == -1:
        return
    try:
        yaml.safe_load(text[3:end])
    except yaml.YAMLError as exc:
        raise PackWriteRejected(
            f"{path}: the frontmatter block is not valid YAML: {exc}",
            path=path,
            line=_yaml_error_line(exc),
        ) from exc


def verify_candidate(
    rel: str, candidate: str, *, pre_image: Optional[str] = None
) -> None:
    """Three checks in order; raises :class:`PackWriteRejected` on failure.

    Called twice: in memory before snapshotting (a refusal must not leave a history
    entry) and on bytes re-read from disk inside :func:`_atomic_write_verified`.
    """
    path = str(rel)
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        loaded = _parse_yaml_or_raise(candidate, path=path)
        if _is_empty_document(loaded) and candidate.strip():
            # Parses but yields nothing: top-level structure missing, loads silently absent.
            raise PackWriteRejected(
                f"{path}: the content is not blank but parses to an empty document — "
                "the top-level key or list is missing",
                path=path,
            )
        if pre_image is not None and _is_empty_document(loaded):
            try:
                before = yaml.safe_load(pre_image)
            except yaml.YAMLError:
                before = None
            if not _is_empty_document(before):
                raise PackWriteRejected(
                    f"{path}: this edit would make the file load as empty, which the "
                    "pack loader cannot tell apart from the file being absent",
                    path=path,
                )
    elif suffix == ".md":
        _parse_frontmatter_or_raise(candidate, path=path)


# --------------------------------------------------------------------- atomic write


def _atomic_write_verified(
    target: Path,
    text: str,
    *,
    rel: str,
    pre_image: Optional[str],
    strict: bool = True,
) -> Dict[str, Any]:
    """Write ``text`` to ``target`` only if what lands on disk verifies.

    PID-suffixed temp, ``fsync``, re-read from disk. ``strict=False`` downgrades for
    :func:`restore` so undo is not blocked by a previously invalid file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
    result: Dict[str, Any] = {"parses": True, "parse_error": ""}
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        written = tmp.read_text(encoding="utf-8")
        try:
            verify_candidate(rel, written, pre_image=pre_image)
        except PackWriteRejected as exc:
            if strict:
                raise
            result["parses"] = False
            result["parse_error"] = str(exc)
            logger.warning("Restoring %s even though it does not parse: %s", rel, exc)
    except PackWriteRejected:
        _unlink_quietly(tmp)
        raise
    except OSError as exc:
        _unlink_quietly(tmp)
        raise PackStoreError(f"{rel}: could not be written: {exc}") from exc
    try:
        os.replace(tmp, target)
    except OSError as exc:
        _unlink_quietly(tmp)
        raise PackStoreError(f"{rel}: could not be swapped into place: {exc}") from exc
    result["bytes"] = len(text.encode("utf-8"))
    result["sha256"] = sha256_text(text)
    result["durable"] = _mirror_push(target)
    return result


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# -------------------------------------------------------------------------- history


def _history_root(pack: str) -> Path:
    return pack_dir(pack) / HISTORY_DIR


def _history_index_path(pack: str) -> Path:
    return _history_root(pack) / "index.json"


def _read_history_index(pack: str) -> List[Dict[str, Any]]:
    path = _history_index_path(pack)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as exc:  # a corrupt index must not block editing
        logger.warning("Unreadable pack history index %s: %s", path, exc)
        return []
    entries = doc.get("entries") if isinstance(doc, dict) else None
    return (
        [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
    )


def _write_history_index(pack: str, entries: List[Dict[str, Any]]) -> None:
    path = _history_index_path(pack)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"entries": entries}, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        with open(tmp, "r", encoding="utf-8") as fh:
            json.load(fh)
        os.replace(tmp, path)
        _mirror_push(path)
    except Exception as exc:
        _unlink_quietly(tmp)
        logger.warning("Could not update pack history index %s: %s", path, exc)


def _blob_path(pack: str, digest: str) -> Path:
    return _history_root(pack) / "blobs" / digest[:2] / digest


def snapshot(
    pack: str,
    rel: str,
    text: str,
    *,
    reason: str = "write",
    actor: str = "",
    session: str = "",
) -> Dict[str, Any]:
    """Store ``text`` content-addressed and append an index entry. Best-effort.

    Content-addressed: edit-then-revert re-uses the first blob. A full disk must not
    block a valid edit; a missing snapshot is reported, not fatal.
    """
    digest = sha256_text(text)
    entry = {
        "id": "",
        "path": str(rel),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": str(reason),
        "actor": str(actor or ""),
        "session": str(session or ""),
        "sha256": digest,
        "bytes": len(text.encode("utf-8")),
        "stored": False,
        # Always present so callers need not distinguish "not durable" from "older entry".
        "durable": True,
    }
    try:
        blob = _blob_path(pack, digest)
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            tmp = blob.with_suffix(f".tmp{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, blob)
            # Written once; push failure is a warning; `stored` flags a missing blob.
            entry["durable"] = _mirror_push(blob)
        else:
            entry["durable"] = True
        entries = _read_history_index(pack)
        entry["id"] = f"snap-{len(entries) + 1:06d}-{digest[:12]}"
        entry["stored"] = True
        entries.append(entry)
        _write_history_index(pack, entries)
    except Exception as exc:
        logger.warning("Could not snapshot %s/%s: %s", pack, rel, exc)
    return entry


def history(pack: str, rel: str = "") -> List[Dict[str, Any]]:
    """Snapshots for one file, or the whole pack. Newest first."""
    entries = _read_history_index(pack)
    if rel:
        wanted = str(safe_rel_path(rel))
        entries = [e for e in entries if e.get("path") == wanted]
    return list(reversed(entries))


def snapshot_text(pack: str, snapshot_id: str) -> str:
    """The stored bytes of one snapshot, for a preview or a diff."""
    entry = _find_snapshot(pack, snapshot_id)
    blob = _blob_path(pack, str(entry.get("sha256", "")))
    if not blob.is_file():
        raise PackFileNotFound(f"the stored content for {snapshot_id} is missing")
    return blob.read_text(encoding="utf-8")


def _find_snapshot(pack: str, snapshot_id: str) -> Dict[str, Any]:
    ident = str(snapshot_id or "").strip()
    for entry in _read_history_index(pack):
        if entry.get("id") == ident:
            return entry
    raise PackFileNotFound(f"unknown snapshot {snapshot_id!r}")


# ----------------------------------------------------------------------- read paths


def list_packs() -> List[Dict[str, Any]]:
    """Every installed pack, with a file count and total size."""
    root = packs_root()
    out: List[Dict[str, Any]] = []
    if not root.is_dir():
        return out
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        if entry.name.startswith("."):
            continue
        try:
            safe_pack_name(entry.name)
        except PackStoreError:
            # Name not addressable; listing it would offer a pack every call refuses.
            logger.debug("Skipping unaddressable pack directory %s", entry.name)
            continue
        files = [n for n in _walk_files(entry)]
        out.append(
            {
                "name": entry.name,
                "files": len(files),
                "bytes": sum(f.stat().st_size for f in files),
            }
        )
    return out


def pack_exists(pack: str) -> bool:
    return pack_dir(pack).is_dir()


def _walk_files(root: Path) -> List[Path]:
    """Every pack file, excluding the history store, dotfiles and mid-write temps."""
    out: List[Path] = []
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if not path.is_file():
            continue
        if _TMP_SUFFIX.search(path.name):
            continue
        out.append(path)
    return out


def pack_tree(pack: str) -> Dict[str, Any]:
    """Flat, sorted node list for the file browser.

    Flat because the browser renders one indented list. ``editable`` is a file
    property: over :data:`INLINE_EDIT_MAX_BYTES` the UI offers a download, not a textarea.
    """
    root = pack_dir(pack)
    if not root.is_dir():
        raise PackNotFound(f"no pack named {pack!r}")
    files = _walk_files(root)
    nodes: List[Dict[str, Any]] = []
    seen_dirs = set()
    for path in files:
        rel = path.relative_to(root)
        for i in range(len(rel.parts) - 1):
            d = "/".join(rel.parts[: i + 1])
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
            nodes.append({"path": d, "dir": True, "depth": i, "name": rel.parts[i]})
        text_bytes = path.stat().st_size
        suffix = path.suffix.lower()
        nodes.append(
            {
                "path": str(PurePosixPath(rel)),
                "dir": False,
                "depth": len(rel.parts) - 1,
                "name": path.name,
                "bytes": text_bytes,
                "lines": _count_lines(path),
                "kind": classify(str(PurePosixPath(rel))),
                "editable": suffix in EDITABLE_SUFFIXES
                and text_bytes <= INLINE_EDIT_MAX_BYTES,
                "text": suffix in EDITABLE_SUFFIXES,
            }
        )
    # Sort by segments, not joined string: "/" sorts before letters in some locales.
    nodes.sort(key=lambda n: tuple(str(n["path"]).split("/")))
    return {
        "pack": safe_pack_name(pack),
        "nodes": nodes,
        "counts": {
            "files": sum(1 for n in nodes if not n["dir"]),
            "dirs": sum(1 for n in nodes if n["dir"]),
            "bytes": sum(int(n.get("bytes") or 0) for n in nodes if not n["dir"]),
        },
    }


def _count_lines(path: Path) -> int:
    if path.suffix.lower() not in EDITABLE_SUFFIXES:
        return 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def read_file(pack: str, rel: str) -> Dict[str, Any]:
    """One file's text plus editing metadata. ``sha256`` is the concurrency token."""
    relp = safe_rel_path(rel)
    path = pack_dir(pack) / relp
    if not path.is_file():
        raise PackFileNotFound(f"{relp}: no such file in pack {pack!r}")
    size = path.stat().st_size
    if size > READ_MAX_BYTES:
        raise PackTooLarge(
            f"{relp}: {size} bytes is over the {READ_MAX_BYTES}-byte read limit — "
            "download it instead"
        )
    require_editable_suffix(relp)
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "pack": safe_pack_name(pack),
        "path": str(relp),
        "kind": classify(str(relp)),
        "text": text,
        "bytes": size,
        "lines": len(text.splitlines()),
        "sha256": sha256_text(text),
        "editable": size <= INLINE_EDIT_MAX_BYTES,
    }


# ---------------------------------------------------------------------- write paths


def _check_write_size(rel: PurePosixPath, text: str) -> None:
    size = len(text.encode("utf-8"))
    if size > WRITE_MAX_BYTES:
        raise PackTooLarge(
            f"{rel}: {size} bytes is over the {WRITE_MAX_BYTES}-byte write limit"
        )


def _current_text(path: Path, rel: PurePosixPath) -> str:
    if not path.is_file():
        raise PackFileNotFound(f"{rel}: no such file")
    return path.read_text(encoding="utf-8", errors="replace")


def _check_expected_sha(rel: PurePosixPath, current: str, expect_sha: str) -> None:
    if not expect_sha:
        return
    actual = sha256_text(current)
    if expect_sha != actual:
        raise PackConflict(
            f"{rel} changed on disk since it was read "
            f"(expected {expect_sha[:12]}, found {actual[:12]}) — reload before saving"
        )


def write_file(
    pack: str,
    rel: str,
    text: str,
    *,
    expect_sha: str = "",
    actor: str = "",
    session: str = "",
    reason: str = "write",
) -> Dict[str, Any]:
    """Replace a file's whole content. The prior bytes are snapshotted first."""
    relp = safe_rel_path(rel)
    require_editable_suffix(relp)
    _check_write_size(relp, text)
    path = pack_dir(pack) / relp
    current = _current_text(path, relp)
    _check_expected_sha(relp, current, expect_sha)
    if text == current:
        return {
            "pack": safe_pack_name(pack),
            "path": str(relp),
            "changed": False,
            "sha256": sha256_text(current),
            "bytes": len(current.encode("utf-8")),
            "snapshot": None,
            # No-op: nothing changed, nothing at risk; reporting non-durable would mislead.
            "durable": True,
        }
    verify_candidate(str(relp), text, pre_image=current)
    snap = snapshot(
        pack, str(relp), current, reason=reason, actor=actor, session=session
    )
    written = _atomic_write_verified(path, text, rel=str(relp), pre_image=current)
    return {
        "pack": safe_pack_name(pack),
        "path": str(relp),
        "changed": True,
        "sha256": written["sha256"],
        "bytes": written["bytes"],
        "lines": len(text.splitlines()),
        "snapshot": snap.get("id") or None,
        # Both halves ANDed: content without its snapshot is unundoable after a restart.
        "durable": bool(written["durable"] and snap.get("durable", True)),
    }


def replace_lines(
    pack: str,
    rel: str,
    start_line: int,
    end_line: int,
    text: str,
    *,
    expect_first_line: str = "",
    expect_last_line: str = "",
    expect_sha: str = "",
    actor: str = "",
    session: str = "",
    reason: str = "write",
) -> Dict[str, Any]:
    """Replace lines ``[start_line, end_line]`` (1-indexed, inclusive).

    Raw-text replacement preserves comments and YAML anchors that a round trip would
    destroy. ``expect_first_line``/``expect_last_line`` detect a stale plan.
    """
    relp = safe_rel_path(rel)
    require_editable_suffix(relp)
    path = pack_dir(pack) / relp
    current = _current_text(path, relp)
    _check_expected_sha(relp, current, expect_sha)
    lines = current.splitlines(keepends=True)
    start, end = int(start_line), int(end_line)
    if start < 1 or end < start:
        raise PackStoreError(
            f"{relp}: bad line range {start_line}-{end_line} (1-indexed, inclusive)"
        )
    if end > len(lines):
        raise PackStoreError(
            f"{relp}: line range {start}-{end} runs past the end of the file "
            f"({len(lines)} lines)"
        )
    if expect_first_line:
        actual = lines[start - 1].rstrip("\n")
        if actual.strip() != expect_first_line.strip():
            raise PackConflict(
                f"{relp}: line {start} is {actual!r}, not {expect_first_line!r} — "
                "the file changed since this edit was computed"
            )
    if expect_last_line:
        actual = lines[end - 1].rstrip("\n")
        if actual.strip() != expect_last_line.strip():
            raise PackConflict(
                f"{relp}: line {end} is {actual!r}, not {expect_last_line!r} — "
                "the file changed since this edit was computed"
            )
    before, after = lines[: start - 1], lines[end:]
    body = text
    if body and not body.endswith("\n"):
        body += "\n"
    if not after and body and not current.endswith("\n"):
        # Range ends the file; don't add a newline the file never had.
        body = body[:-1]
    candidate = "".join(before) + body + "".join(after)
    _check_write_size(relp, candidate)
    if candidate == current:
        return {
            "pack": safe_pack_name(pack),
            "path": str(relp),
            "changed": False,
            "sha256": sha256_text(current),
            "bytes": len(current.encode("utf-8")),
            "snapshot": None,
            "durable": True,  # nothing changed; see write_file
        }
    verify_candidate(str(relp), candidate, pre_image=current)
    snap = snapshot(
        pack, str(relp), current, reason=reason, actor=actor, session=session
    )
    written = _atomic_write_verified(path, candidate, rel=str(relp), pre_image=current)
    return {
        "pack": safe_pack_name(pack),
        "path": str(relp),
        "changed": True,
        "sha256": written["sha256"],
        "bytes": written["bytes"],
        "lines": len(candidate.splitlines()),
        "replaced": [start, end],
        "snapshot": snap.get("id") or None,
        "durable": bool(written["durable"] and snap.get("durable", True)),
    }


def create_file(
    pack: str,
    rel: str,
    text: str = "",
    *,
    actor: str = "",
    session: str = "",
) -> Dict[str, Any]:
    """Create a file that does not exist yet. Separate from :func:`write_file`
    so a save on a mistyped path does not silently create a new file."""
    relp = safe_rel_path(rel)
    require_editable_suffix(relp)
    _check_write_size(relp, text)
    path = pack_dir(pack) / relp
    if not pack_dir(pack).is_dir():
        raise PackNotFound(f"no pack named {pack!r}")
    if path.exists():
        raise PackFileExists(f"{relp}: already exists — save it instead")
    written = _atomic_write_verified(path, text, rel=str(relp), pre_image=None)
    snap = snapshot(
        pack, str(relp), text, reason="create", actor=actor, session=session
    )
    return {
        "pack": safe_pack_name(pack),
        "path": str(relp),
        "created": True,
        "sha256": written["sha256"],
        "bytes": written["bytes"],
        "lines": len(text.splitlines()),
        "kind": classify(str(relp)),
        "snapshot": snap.get("id") or None,
        "durable": bool(written["durable"] and snap.get("durable", True)),
    }


def delete_file(
    pack: str,
    rel: str,
    *,
    actor: str = "",
    session: str = "",
) -> Dict[str, Any]:
    """Remove a file, keeping its content in history. Snapshot taken before unlink."""
    relp = safe_rel_path(rel)
    path = pack_dir(pack) / relp
    if not path.is_file():
        raise PackFileNotFound(f"{relp}: no such file")
    text = path.read_text(encoding="utf-8", errors="replace")
    snap = snapshot(
        pack, str(relp), text, reason="delete", actor=actor, session=session
    )
    if not snap.get("stored"):
        raise PackStoreError(
            f"{relp}: refusing to delete — its content could not be saved to history, "
            "so the delete would not be reversible"
        )
    path.unlink()
    # Must be recorded durably: the bundle ships this file and re-seeds it on a boot with
    # no record of the delete.
    recorded = _mirror_delete(path)
    return {
        "pack": safe_pack_name(pack),
        "path": str(relp),
        "deleted": True,
        "bytes": len(text.encode("utf-8")),
        "snapshot": snap.get("id"),
        "durable": bool(recorded and snap.get("durable", True)),
    }


def restore(
    pack: str,
    snapshot_id: str,
    *,
    rel: str = "",
    actor: str = "",
    session: str = "",
) -> Dict[str, Any]:
    """Put a snapshot's content back, snapshotting the current content first.

    The restore is itself undoable. Verification is downgraded (``strict=False``) so
    undo is not blocked by a previously invalid file.
    """
    entry = _find_snapshot(pack, snapshot_id)
    target_rel = str(safe_rel_path(rel)) if rel else str(entry.get("path") or "")
    if not target_rel:
        raise PackStoreError(f"snapshot {snapshot_id!r} names no file")
    relp = safe_rel_path(target_rel)
    require_editable_suffix(relp)
    text = snapshot_text(pack, snapshot_id)
    path = pack_dir(pack) / relp
    existed = path.is_file()
    if existed:
        current = path.read_text(encoding="utf-8", errors="replace")
        snapshot(
            pack, str(relp), current, reason="restore", actor=actor, session=session
        )
    written = _atomic_write_verified(
        path, text, rel=str(relp), pre_image=None, strict=False
    )
    return {
        "pack": safe_pack_name(pack),
        "path": str(relp),
        "restored": snapshot_id,
        "recreated": not existed,
        "sha256": written["sha256"],
        "bytes": written["bytes"],
        "parses": written["parses"],
        "parse_error": written["parse_error"],
        "durable": bool(written["durable"]),
    }


# ------------------------------------------------------------------------- scaffold


def _vocabulary_file(words: List[str]) -> str:
    """Template header followed by the caller's own words. Header kept verbatim:
    it is the only copy of the omission rule."""
    header = ""
    src = TEMPLATE_DIR / "domain_vocabulary.yaml"
    if src.is_file():
        raw = src.read_text(encoding="utf-8")
        cut = raw.find("\ndomain_vocabulary:")
        if cut != -1:
            header = raw[: cut + 1]
    lines = [header] if header else []
    lines.append("domain_vocabulary:")
    for word in words:
        lines.append(f"  - {word}")
    return "\n".join(lines) + "\n"


def normalise_vocabulary(vocabulary: Any) -> List[str]:
    """Validate and de-duplicate a word list, or raise. Empty is refused: a pack
    without ``domain_vocabulary.yaml`` fails the neutrality suite."""
    if isinstance(vocabulary, str):
        raw = [vocabulary]
    elif isinstance(vocabulary, (list, tuple)):
        raw = list(vocabulary)
    else:
        raw = []
    words: List[str] = []
    for item in raw:
        word = str(item or "").strip().lower()
        if not word:
            continue
        if not _VOCAB_WORD.fullmatch(word):
            raise PackStoreError(f"invalid vocabulary word {item!r}")
        if word not in words:
            words.append(word)
    if not words:
        raise PackStoreError(
            "a new pack must declare at least one word in domain_vocabulary.yaml — "
            "it is what makes the engine provably free of this domain's vocabulary"
        )
    return words


def scaffold_pack(
    name: str, *, vocabulary: Any, actor: str = "", session: str = ""
) -> Dict[str, Any]:
    """Create a new pack from the checked-in template, minus :data:`_TEMPLATE_SKIP`.
    ``domain_vocabulary.yaml`` is written from ``vocabulary``."""
    pack = safe_pack_name(name)
    words = normalise_vocabulary(vocabulary)
    target = knowledge_pack_dir(pack)
    if target.exists():
        raise PackFileExists(f"a pack named {pack!r} already exists")
    if not TEMPLATE_DIR.is_dir():
        raise PackStoreError(
            f"the pack template is missing from {TEMPLATE_DIR} — cannot scaffold"
        )
    copied: List[str] = []
    durable = True
    target.mkdir(parents=True)
    try:
        for src in _walk_files(TEMPLATE_DIR):
            rel = PurePosixPath(src.relative_to(TEMPLATE_DIR))
            if str(rel) in _TEMPLATE_SKIP or rel.name in _TEMPLATE_SKIP:
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            # Push: a scaffolded pack is not in the bundle and won't be re-seeded.
            durable = _mirror_push(dest) and durable
            copied.append(str(rel))
        vocab = _vocabulary_file(words)
        written = _atomic_write_verified(
            target / "domain_vocabulary.yaml",
            vocab,
            rel="domain_vocabulary.yaml",
            pre_image=None,
        )
        durable = bool(written["durable"]) and durable
        copied.append("domain_vocabulary.yaml")
    except Exception:
        # Roll back: a half-copied pack loads with pieces missing.
        shutil.rmtree(target, ignore_errors=True)
        raise
    snapshot(
        pack,
        "domain_vocabulary.yaml",
        vocab,
        reason="create",
        actor=actor,
        session=session,
    )
    return {
        "pack": pack,
        "created": True,
        "files": sorted(copied),
        "vocabulary": words,
        "template": str(TEMPLATE_DIR.relative_to(REPO_ROOT)),
        "durable": durable,
    }


def unified_diff(before: str, after: str, path: str) -> str:
    """Unified diff, server-side. The diff is the approval basis and must match
    what is actually written."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
