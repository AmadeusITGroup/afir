"""One caller's edits to a shared tree, and what happens to them when the shared tree moves.

Both editable trees work the same way: an administrator edits the **base** — the working copy
`config/` and `knowledge/<pack>/`, mirrored exactly as before — and everyone else edits their
own **layer**. A layer holds only the files that caller changed, for the reason
`storage.mirror` holds edits rather than a tree: a layer that carried the whole tree would
silently override the next release.

When the base moves under a layer, each layered file is re-merged against it. The merge base
is the text the caller forked from, kept beside their edit: `.history` also holds it, but a
history blob can be pruned, and a missing merge base turns a clean merge into a conflict the
caller did not cause.
"""

import json
import logging
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from src.identity import USER_PREFIX, storage_segment

logger = logging.getLogger(__name__)

#: Under the per-caller namespace, beside `jobs/` and `exports/`.
LAYER_ROOT = "layers"

#: The two trees a caller may layer over. Names, not paths: the layer is keyed by the same
#: relative path the base tree uses, so a merge never has to translate between them.
CONFIG_LAYER = "config"
KNOWLEDGE_LAYER = "knowledge"

#: Suffix of the blob holding the base a layered file was forked from.
BASE_SUFFIX = ".base"

#: Suffix of the blob holding one layered file's metadata.
META_SUFFIX = ".meta.json"

#: How a rebase turned out. `conflict` still keeps the caller's text — losing an edit is
#: worse than keeping one that no longer applies cleanly.
CLEAN = "clean"
MERGED = "merged"
CONFLICT = "conflict"
ADOPTED = "adopted"

CONFLICT_MINE = "<<<<<<< your version"
CONFLICT_SPLIT = "======="
CONFLICT_THEIRS = ">>>>>>> the administrator's version"


class UserLayer:
    """One caller's edits to one tree. Best-effort, like every storage caller here."""

    def __init__(self, storage, segment: str, label: str):
        self._storage = storage
        self.segment = storage_segment(segment)
        self.label = str(label)

    @property
    def available(self) -> bool:
        return self._storage is not None

    def _key(self, rel: str, suffix: str = "") -> str:
        return f"{USER_PREFIX}/{self.segment}/{LAYER_ROOT}/{self.label}/{rel}{suffix}"

    # -- read --------------------------------------------------------------

    def read(self, rel: str) -> Optional[str]:
        """This caller's version of `rel`, or None when they have not touched it."""
        if not self.available:
            return None
        try:
            return self._storage.get_text(self._key(rel))
        except Exception as exc:  # noqa: BLE001 — a layer must never fail a read
            logger.warning("Could not read layer %s/%s: %s", self.label, rel, exc)
            return None

    def base_of(self, rel: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            return self._storage.get_text(self._key(rel, BASE_SUFFIX))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read merge base %s/%s: %s", self.label, rel, exc)
            return None

    def meta_of(self, rel: str) -> dict:
        if not self.available:
            return {}
        try:
            raw = self._storage.get_text(self._key(rel, META_SUFFIX))
        except Exception:  # noqa: BLE001
            return {}
        try:
            loaded = json.loads(raw) if raw else {}
        except ValueError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def paths(self) -> List[str]:
        """Every relative path this caller has layered, in listing order."""
        if not self.available:
            return []
        root = f"{USER_PREFIX}/{self.segment}/{LAYER_ROOT}/{self.label}"
        cut = len(root) + 1
        out = []
        try:
            objects = self._storage.list_keys(root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list layer %s: %s", self.label, exc)
            return []
        for obj in objects:
            key = obj.key
            if key.endswith(BASE_SUFFIX) or key.endswith(META_SUFFIX):
                continue
            if len(key) > cut:
                out.append(key[cut:])
        return sorted(out)

    def describe(self) -> List[dict]:
        """One row per layered file, for the tab that has to show what is overridden."""
        rows = []
        for rel in self.paths():
            meta = self.meta_of(rel)
            rows.append(
                {
                    "path": rel,
                    "state": meta.get("state", CLEAN),
                    "edited_at": meta.get("edited_at"),
                    "rebased_at": meta.get("rebased_at"),
                    "conflict": meta.get("state") == CONFLICT,
                }
            )
        return rows

    # -- write -------------------------------------------------------------

    def write(self, rel: str, text: str, base_text: Optional[str]) -> bool:
        """Record this caller's version of `rel`, forked from `base_text`.

        `base_text` is stored on the first write only: a caller editing their own override
        twice has still forked from the same base, and overwriting it with their own previous
        text would make every later rebase a no-op.
        """
        if not self.available:
            return False
        try:
            ok = self._storage.put_text(self._key(rel), text)
            if ok and self._storage.get_text(self._key(rel, BASE_SUFFIX)) is None:
                self._storage.put_text(self._key(rel, BASE_SUFFIX), base_text or "")
            self._write_meta(rel, state=CLEAN, edited_at=_now())
            return ok
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write layer %s/%s: %s", self.label, rel, exc)
            return False

    def drop(self, rel: str) -> bool:
        """Discard this caller's override so they see the base again."""
        if not self.available:
            return False
        ok = False
        for suffix in ("", BASE_SUFFIX, META_SUFFIX):
            try:
                ok = self._storage.delete(self._key(rel, suffix)) or ok
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not drop layer %s/%s: %s", self.label, rel, exc)
        return ok

    def _write_meta(self, rel: str, **fields) -> None:
        meta = self.meta_of(rel)
        meta.update({k: v for k, v in fields.items() if v is not None})
        try:
            self._storage.put_text(
                self._key(rel, META_SUFFIX), json.dumps(meta, indent=2)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write layer meta %s/%s: %s", self.label, rel, exc)

    # -- rebase ------------------------------------------------------------

    def rebase(self, rel: str, new_base: Optional[str]) -> str:
        """Re-merge this caller's override onto a moved base. Returns one of the states.

        Their edit is never discarded. A base that has been deleted leaves the override in
        place and says so, rather than deleting a caller's work to match an absence.
        """
        mine = self.read(rel)
        if mine is None:
            return CLEAN
        if new_base is None:
            self._write_meta(rel, state=CLEAN, rebased_at=_now())
            return CLEAN
        base = self.base_of(rel)
        if base is None:
            # No fork point: their text is all that is known, so it stands as an adoption
            # rather than being merged against a base it may never have come from.
            self._write_meta(rel, state=ADOPTED, rebased_at=_now())
            return ADOPTED
        if mine == new_base or mine == base:
            # Either the release caught up with the caller, or the caller never actually
            # changed anything. Both make the override dead weight that would otherwise
            # keep pinning this file against every future release.
            self.drop(rel)
            return CLEAN
        if base == new_base:
            return CLEAN
        merged, conflicted = merge_three_way(base, mine, new_base)
        state = CONFLICT if conflicted else MERGED
        try:
            self._storage.put_text(self._key(rel), merged)
            self._storage.put_text(self._key(rel, BASE_SUFFIX), new_base)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not rebase layer %s/%s: %s", self.label, rel, exc)
            return CONFLICT
        self._write_meta(rel, state=state, rebased_at=_now())
        return state


def merge_three_way(base: str, mine: str, theirs: str) -> Tuple[str, bool]:
    """Line-based three-way merge. Returns (text, conflicted).

    One side's change to a region wins where the other left that region alone; a region both
    changed differently is marked rather than resolved, because picking one silently is how a
    caller's edit or an administrator's release disappears.
    """
    base_lines = base.splitlines(keepends=True)
    mine_edits = _edits(base_lines, mine.splitlines(keepends=True))
    their_edits = _edits(base_lines, theirs.splitlines(keepends=True))

    out: List[str] = []
    conflicted = False
    position = 0
    for start, end, mine_text, their_text in _align(mine_edits, their_edits):
        out.extend(base_lines[position:start])
        position = end
        if their_text is None:
            out.extend(mine_text or [])
        elif mine_text is None:
            out.extend(their_text)
        elif mine_text == their_text:
            out.extend(mine_text)
        else:
            conflicted = True
            out.extend(_conflict_block(mine_text, their_text))
    out.extend(base_lines[position:])
    return "".join(out), conflicted


def _edits(base_lines: Sequence[str], other_lines: Sequence[str]):
    """Non-equal opcodes as (start, end, replacement) over base line indices."""
    matcher = SequenceMatcher(None, list(base_lines), list(other_lines), autojunk=False)
    return [
        (i1, i2, list(other_lines[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def _align(mine_edits, their_edits):
    """Walk both edit lists together, yielding (start, end, mine|None, theirs|None).

    Overlapping regions are coalesced into one span so a conflict is reported once over the
    whole disputed area rather than interleaved line by line.
    """
    spans = sorted(
        [(s, e, "mine", text) for s, e, text in mine_edits]
        + [(s, e, "theirs", text) for s, e, text in their_edits]
    )
    index = 0
    while index < len(spans):
        start, end, side, text = spans[index]
        group = [(side, start, end, text)]
        index += 1
        # An insertion is a zero-width span; it only groups with a real overlap, never with
        # an edit that merely starts where it sits.
        while index < len(spans) and spans[index][0] < end:
            nside, nend = spans[index][2], spans[index][1]
            group.append((nside, spans[index][0], nend, spans[index][3]))
            end = max(end, nend)
            index += 1
        mine_text = _side(group, "mine")
        their_text = _side(group, "theirs")
        yield start, end, mine_text, their_text


def _side(group, want) -> Optional[List[str]]:
    parts = [text for side, _s, _e, text in group if side == want]
    if not parts:
        return None
    return [line for part in parts for line in part]


def _conflict_block(mine_text, their_text) -> List[str]:
    return (
        [CONFLICT_MINE + "\n"]
        + _terminated(mine_text)
        + [CONFLICT_SPLIT + "\n"]
        + _terminated(their_text)
        + [CONFLICT_THEIRS + "\n"]
    )


def _terminated(lines: Sequence[str]) -> List[str]:
    """Ensure a marker starts on its own line, even where the region had no final newline."""
    out = list(lines)
    if out and not out[-1].endswith("\n"):
        out[-1] = out[-1] + "\n"
    return out


def splice_lines(text: str, start_line: int, end_line: int, body: str) -> str:
    """Replace lines ``[start_line, end_line]`` (1-indexed, inclusive) in `text`.

    The same arithmetic ``pack_store.replace_lines`` performs against a file, over text, so a
    caller editing their own layer gets the anchored line replacement rather than being pushed
    to a whole-file save that would destroy the comments the range editor exists to preserve.
    Raises ``ValueError`` on a range the text cannot satisfy — a silently clamped range writes
    the edit somewhere the author did not point at.
    """
    lines = text.splitlines(keepends=True)
    start, end = int(start_line), int(end_line)
    if start < 1 or end < start:
        raise ValueError(f"bad line range {start_line}-{end_line} (1-indexed, inclusive)")
    if end > len(lines):
        raise ValueError(
            f"line range {start}-{end} runs past the end of the file ({len(lines)} lines)"
        )
    after = lines[end:]
    if body and not body.endswith("\n"):
        body += "\n"
    if not after and body and not text.endswith("\n"):
        # The range ends the file; don't add a newline the file never had.
        body = body[:-1]
    return "".join(lines[: start - 1]) + body + "".join(after)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))


class OverlaySet:
    """The two layers one caller has, built together so a handler asks for one thing."""

    def __init__(self, storage, segment: str):
        self.config = UserLayer(storage, segment, CONFIG_LAYER)
        self.knowledge = UserLayer(storage, segment, KNOWLEDGE_LAYER)

    def layer(self, label: str) -> UserLayer:
        return self.config if label == CONFIG_LAYER else self.knowledge


def rebase_all(storage, label: str, base_reader) -> Dict[str, Dict[str, str]]:
    """Re-merge every caller's layer of `label` after an administrator moved the base.

    `base_reader(rel)` returns the new base text, or None where the file is gone. Returns
    `{segment: {path: state}}` for only the files whose state is worth reporting — a clean
    rebase is not news, a conflict is.
    """
    report: Dict[str, Dict[str, str]] = {}
    for segment in _segments(storage):
        layer = UserLayer(storage, segment, label)
        for rel in layer.paths():
            state = layer.rebase(rel, base_reader(rel))
            if state in (MERGED, CONFLICT, ADOPTED):
                report.setdefault(segment, {})[rel] = state
    return report


def _segments(storage) -> List[str]:
    """Every caller with anything stored. Derived from the store, so it needs no registry."""
    if storage is None:
        return []
    try:
        objects = storage.list_keys(USER_PREFIX)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list per-caller storage: %s", exc)
        return []
    found = set()
    for obj in objects:
        parts = obj.key.split("/")
        if len(parts) > 2 and parts[0] == USER_PREFIX:
            found.add(parts[1])
    return sorted(found)
