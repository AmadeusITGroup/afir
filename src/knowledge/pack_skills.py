"""The pack-authoring skill library: method knowledge selected and injected for the assistant.

``skills/*.md`` holds technique, not domain facts. That is mechanical rather than a promise:
:func:`validate_pack`'s neutrality check scans every ``.md`` under ``src/`` — this directory
included — against the validated pack's own vocabulary.

Selection is deterministic: :func:`select` matches by triggers and injects; :func:`read` is the
widening path. A skill that fails to parse is reported (not dropped) via :func:`problems`.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)

#: Beside this module, not ``docs/``: the bundle must ship assistant and library together.
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

#: Cap including always-on skills. A question with many keywords must not crowd out the pack's own files.
MAX_INJECTED_SKILLS = 4

#: Skill text budget per turn before the model has read any pack file.
INJECTED_CHARS_BUDGET = 32_000

#: Per-call cap on :func:`read`; marked in the returned text when it bites, never silent.
READ_CHARS_PER_CALL = 16_000


@dataclass
class Skill:
    """One method document. ``triggers`` and ``always`` are what make selection mechanical."""

    name: str
    title: str
    when: str
    body: str
    always: bool = False
    triggers: List[str] = field(default_factory=list)
    path: str = ""

    @property
    def chars(self) -> int:
        return len(self.body)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "when": self.when,
            "always": self.always,
            "chars": self.chars,
        }


# --------------------------------------------------------------------------- loading

#: ``(signature, skills, problems)``. Signature-keyed so edits are picked up without restart.
_CACHE: Dict[str, Tuple[Any, List[Skill], List[str]]] = {}

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)


def _signature(directory: Path) -> Any:
    try:
        return tuple(
            sorted(
                (p.name, p.stat().st_mtime_ns, p.stat().st_size)
                for p in directory.glob("*.md")
            )
        )
    except OSError:
        return ()


def _parse(path: Path) -> Tuple[Optional[Skill], str]:
    """One file to a :class:`Skill`, or ``(None, problem)`` — never raises.

    A malformed skill is skipped and reported, not raised: silently missing it is worse
    than having one fewer.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return None, f"{path.name}: unreadable ({e.__class__.__name__})"
    match = _FRONTMATTER.match(text)
    if not match:
        return None, f"{path.name}: no `---` frontmatter block"
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        return None, f"{path.name}: frontmatter does not parse ({e.__class__.__name__})"
    if not isinstance(meta, dict):
        return None, f"{path.name}: frontmatter is not a mapping"
    name = str(meta.get("name") or "").strip()
    if not name:
        return None, f"{path.name}: frontmatter declares no `name`"
    body = match.group(2).strip()
    if not body:
        return None, f"{path.name}: has frontmatter and no body"
    raw_triggers = meta.get("triggers") or []
    if not isinstance(raw_triggers, (list, tuple)):
        return None, f"{path.name}: `triggers` must be a list"
    triggers = [str(t).strip().lower() for t in raw_triggers if str(t).strip()]
    return (
        Skill(
            name=name,
            title=str(meta.get("title") or name),
            when=str(meta.get("when") or ""),
            body=body,
            always=bool(meta.get("always")),
            triggers=triggers,
            path=path.name,
        ),
        "",
    )


def load(directory: Optional[Path] = None) -> List[Skill]:
    """Every parseable skill, ordered always-on first then by file name (so, stably)."""
    return _load_both(directory)[0]


def problems(directory: Optional[Path] = None) -> List[str]:
    """What could not be loaded, one line each. Empty is the normal case."""
    return _load_both(directory)[1]


def _load_both(directory: Optional[Path] = None) -> Tuple[List[Skill], List[str]]:
    root = Path(directory) if directory else SKILLS_DIR
    key = str(root)
    signature = _signature(root)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]
    skills: List[Skill] = []
    issues: List[str] = []
    if not root.is_dir():
        # Missing is not an error: no library is the pre-library behaviour.
        _CACHE[key] = (signature, skills, issues)
        return skills, issues
    for path in sorted(root.glob("*.md")):
        skill, problem = _parse(path)
        if skill is None:
            logger.warning("Skill %s could not be loaded: %s", path.name, problem)
            issues.append(problem)
            continue
        skills.append(skill)
    seen: Dict[str, str] = {}
    for skill in skills:
        if skill.name in seen:
            issues.append(
                f"{skill.path}: duplicate skill name {skill.name!r} "
                f"(also in {seen[skill.name]})"
            )
        seen[skill.name] = skill.path
    skills.sort(key=lambda s: (not s.always, s.path))
    _CACHE[key] = (signature, skills, issues)
    return skills, issues


def by_name(name: str, directory: Optional[Path] = None) -> Optional[Skill]:
    wanted = str(name or "").strip().lower()
    for skill in load(directory):
        if skill.name.lower() == wanted:
            return skill
    return None


# ------------------------------------------------------------------------- selection


def _trigger_pattern(trigger: str) -> re.Pattern:
    """Word-edge pattern where ``_`` is a separator: ``filter`` matches ``never_filter``,
    ``cap`` does not match ``capture``."""
    return re.compile(
        r"(?<![a-z0-9])" + re.escape(trigger.lower()) + r"(?![a-z0-9])", re.I
    )


def select(
    question: str,
    *,
    focus: Sequence[str] = (),
    limit: int = MAX_INJECTED_SKILLS,
    char_budget: int = INJECTED_CHARS_BUDGET,
    directory: Optional[Path] = None,
) -> List[Tuple[Skill, List[str]]]:
    """The skills this question needs, most relevant first, with the matched triggers.

    Always-on skills come first unconditionally. Others rank by distinct trigger count;
    ties break on file order. ``focus`` paths are matched alongside the question text.
    """
    haystack = " ".join([str(question or "")] + [str(f) for f in focus]).lower()
    # Tier 0 = always-on (not a score): scored, it would lose to two-trigger questions and `limit` drops the spine.
    ranked: List[Tuple[int, int, int, Skill, List[str]]] = []
    for order, skill in enumerate(load(directory)):
        if skill.always:
            ranked.append((0, 0, order, skill, []))
            continue
        matched = [t for t in skill.triggers if _trigger_pattern(t).search(haystack)]
        if matched:
            ranked.append((1, -len(matched), order, skill, matched))
    ranked.sort(key=lambda row: row[:3])

    out: List[Tuple[Skill, List[str]]] = []
    spent = 0
    for _, _, _, skill, matched in ranked:
        if len(out) >= max(1, int(limit)):
            break
        # Always-on admitted even if it exceeds the budget: a budget-removable spine is no spine.
        if spent + skill.chars > char_budget and not skill.always:
            continue
        out.append((skill, matched))
        spent += skill.chars
    return out


# ------------------------------------------------------------------- rendering for a prompt


def index_text(directory: Optional[Path] = None) -> str:
    """The catalogue, for the system prompt: one line per skill, name and when to read it."""
    skills = load(directory)
    if not skills:
        return ""
    lines = [
        f"  {s.name} — {s.when or s.title}" + ("  [included below]" if s.always else "")
        for s in skills
    ]
    return "\n".join(lines)


def injected_text(selected: Sequence[Tuple[Skill, List[str]]]) -> str:
    """The selected skills as one block of prompt text, each labelled with why it is here."""
    if not selected:
        return ""
    parts = [
        "METHOD KNOWLEDGE FOR THIS CHANGE. These are this project's own hard-won rules for "
        "authoring a pack, selected for what you were asked. They describe TECHNIQUE, not this "
        "domain — they outrank your priors about how such a change is normally made, and they "
        "are subordinate to what the pack's own files actually say."
    ]
    for skill, matched in selected:
        why = (
            "always applies"
            if skill.always
            else "matched: " + ", ".join(sorted(matched)[:6])
        )
        parts.append(f"--- skill: {skill.name} ({why}) ---\n{skill.body}")
    return "\n\n".join(parts)


def read(name: str, directory: Optional[Path] = None) -> str:
    """One skill's full text, for the ``read_skill`` tool. Bounded, and says when it was cut."""
    skill = by_name(name, directory)
    if skill is None:
        available = ", ".join(s.name for s in load(directory)) or "none installed"
        return f"there is no skill called {name!r}. Available: {available}"
    body = skill.body
    if len(body) > READ_CHARS_PER_CALL:
        return (
            f"skill: {skill.name} — {skill.title}\n"
            f"TRUNCATED at {READ_CHARS_PER_CALL} of {len(body)} characters\n\n"
            + body[:READ_CHARS_PER_CALL]
        )
    return f"skill: {skill.name} — {skill.title}\n\n{body}"


def tool_schema(directory: Optional[Path] = None) -> Dict[str, Any]:
    """``read_skill`` tool definition. Names in the enum so a wrong name fails client-side."""
    names = [s.name for s in load(directory)]
    return {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": (
                "Read one of this project's pack-authoring method documents in full. The ones "
                "relevant to your question were already included in the request; use this for "
                "another one the work turns out to touch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "the skill name",
                        **({"enum": names} if names else {}),
                    }
                },
                "required": ["name"],
            },
        },
    }
