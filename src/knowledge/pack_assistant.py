"""
The knowledge-pack assistant: explore a pack read-only, then propose a set of edits.

The assistant has no write tool: the only path from a model output to disk is
:func:`apply_plan`, called by an HTTP handler after a human approved the plan. Two
budgets bound the loop: :data:`MAX_TURNS` (a model still reading never proposes) and
:data:`MAX_TOOL_BYTES` (large generated files exhaust any context). Exhausting either
appends a "propose from what you have" turn rather than stopping silently.

A plan carries pre-image anchors that :func:`plan_preview` checks at apply time.
Sessions are in-memory and capped (:data:`MAX_SESSIONS`): a persisted plan applied to
files that moved is dangerous. An unknown ``op`` is skipped and reported; any other
validation failure blocks the whole apply, because a half-applied change leaves the
pack in a state neither the before nor the after explains.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from src.knowledge import (
    pack_attachments,
    pack_dry_run,
    pack_skills,
    pack_store,
    pack_validate,
)

logger = logging.getLogger(__name__)

#: Exploration turns before the model is told to propose from what it has. Eight is
#: enough to read a handful of files and search twice; a model still reading at that
#: point is not converging.
MAX_TURNS = 8

#: Total bytes every tool result may return, across the whole session.
MAX_TOOL_BYTES = 200_000

#: Per-call cap on ``read_file``. Marked in the returned text, never silently applied;
#: a model that cannot tell it received a fragment will propose against the fragment.
READ_CHARS_PER_CALL = 40_000

#: Hits ``search`` returns at most.
MAX_SEARCH_HITS = 40

#: Files ``list_files`` returns at most.
MAX_LIST_FILES = 400

#: Live sessions kept. A plan is only meaningful against the snapshot it was computed
#: from, so there is nothing to gain from keeping more.
MAX_SESSIONS = 20

#: Characters of a single focus file pre-loaded in the no-tool fallback.
FALLBACK_FILE_CHARS = 24_000

#: Total characters of pre-loaded context in the no-tool fallback.
FALLBACK_TOTAL_CHARS = 90_000


# ------------------------------------------------------------------ the output models


class EditOp(BaseModel):
    """One file change. This is the LLM's output schema, so its fields are its contract.

    ``patch`` replaces the inclusive 1-indexed range ``[start_line, end_line]`` and is
    what keeps a pack's comments and YAML anchors: one of the installed rulesets is about
    60% comments, and a whole-file rewrite loses every one of them along with the anchors
    the catalogue depends on. ``create`` carries the whole file in ``text``.

    The two ``expect_*`` fields are the anchors. They are what makes a line number safe to
    act on later: the plan was computed against a snapshot, and by apply time the file may
    have moved.
    """

    op: str = Field(default="patch", description="create, patch or delete")
    path: str = Field(
        default="", description="path inside the pack, e.g. shared/checks/x.yaml"
    )
    text: str = Field(
        default="", description="create: the whole file; patch: the replacement lines"
    )
    start_line: int = Field(
        default=0, description="1-indexed inclusive first line to replace"
    )
    end_line: int = Field(
        default=0, description="1-indexed inclusive last line to replace"
    )
    expect_first_line: str = Field(
        default="", description="the line currently at start_line"
    )
    expect_last_line: str = Field(
        default="", description="the line currently at end_line"
    )
    reason: str = Field(default="", description="why this change, in one sentence")


class EditPlan(BaseModel):
    """What the assistant returns: a summary, the ops, and what it could not settle.

    ``questions`` is not decoration. The alternative to an assistant that says "I could not
    determine which ruleset should import this" is one that guesses and produces a
    confident-looking op; a guess inside an approved diff is indistinguishable from a
    finding.
    """

    summary: str = Field(
        default="", description="what this change does, for the operator"
    )
    ops: List[EditOp] = Field(default_factory=list)
    notes: List[str] = Field(
        default_factory=list, description="facts worth stating about the change"
    )
    questions: List[str] = Field(
        default_factory=list, description="what could not be determined from the pack"
    )


# ------------------------------------------------------------------------- the session


@dataclass
class AssistSession:
    """One request's whole life: the trail, the plan, and the events the UI replays."""

    id: str
    pack: str
    question: str
    status: str = "queued"
    tool_mode: str = "tools"
    image_mode: str = "none"
    turns: int = 0
    tool_bytes: int = 0
    budget_spent: str = ""
    plan: Optional[EditPlan] = None
    preview: Optional[Dict[str, Any]] = None
    error: str = ""
    trail: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    focus: List[str] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    #: Which method skills were injected, and what they matched on. On the snapshot because a
    #: plan written with the probe-before-declare rules in front of the model and one written
    #: without them are different artifacts, and the diff does not say which this is.
    skills: List[Dict[str, Any]] = field(default_factory=list)
    #: The rendered skill block. Held so the images probe can rebuild the opening turn without
    #: re-selecting (selection is pure, but its note must be emitted exactly once).
    skill_text: str = field(default="", repr=False)
    #: The running task, parked here only so it stays referenced; asyncio holds a weak
    #: reference to a task nobody awaits, and a garbage collection mid-exploration would
    #: leave a session stuck on "exploring" forever with no error to report.
    task: Any = field(default=None, repr=False)
    _queues: List[asyncio.Queue] = field(default_factory=list, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session": self.id,
            "pack": self.pack,
            "question": self.question,
            "status": self.status,
            "tool_mode": self.tool_mode,
            "image_mode": self.image_mode,
            "turns": self.turns,
            "tool_bytes": self.tool_bytes,
            "budget_spent": self.budget_spent,
            "skills": list(self.skills),
            "error": self.error,
            "trail": list(self.trail),
            "plan": self.plan.model_dump() if self.plan else None,
            "preview": self.preview,
            # An allowlist, so neither an image's raw bytes nor a document's extracted text
            # rides back out on every poll. The operator needs to know which files were used
            # and whether any was cut (`note` carries that), not to re-download them.
            "attachments": [
                {k: a[k] for k in ("name", "bytes", "kind", "chars", "note") if k in a}
                for a in self.attachments
            ],
            "created_at": self.created_at,
        }


#: Live sessions, oldest first so the cap evicts the oldest.
_SESSIONS: "OrderedDict[str, AssistSession]" = OrderedDict()
_SESSION_SEQ = [0]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session(pack: str, question: str) -> AssistSession:
    """Register a session, evicting the oldest once over the cap."""
    _SESSION_SEQ[0] += 1
    ident = f"assist-{_SESSION_SEQ[0]:05d}"
    session = AssistSession(id=ident, pack=pack, question=question, created_at=_now())
    _SESSIONS[ident] = session
    while len(_SESSIONS) > MAX_SESSIONS:
        evicted, _ = _SESSIONS.popitem(last=False)
        logger.debug("Evicted assist session %s", evicted)
    return session


def get_session(session_id: str) -> Optional[AssistSession]:
    return _SESSIONS.get(str(session_id or ""))


def list_sessions(pack: str = "") -> List[Dict[str, Any]]:
    out = [s.snapshot() for s in reversed(list(_SESSIONS.values()))]
    if pack:
        out = [s for s in out if s["pack"] == pack]
    # The plan and the trail are the bulk of a snapshot and are useless in a list; the
    # index only has to say which sessions exist and how they ended.
    for entry in out:
        entry.pop("plan", None)
        entry.pop("trail", None)
        entry.pop("preview", None)
    return out


def emit(session: AssistSession, kind: str, **data) -> Dict[str, Any]:
    """Record an event and hand it to every live subscriber.

    Buffered as well as pushed, so a UI that subscribes late still sees the whole trail;
    the same reason ``JobManager.subscribe`` replays history first. A tool round that
    happened but is not visible reads, to the operator, as a model that did nothing.
    """
    event = {"type": kind, "at": _now(), "session": session.id, **data}
    session.events.append(event)
    for queue in list(session._queues):
        try:
            queue.put_nowait(event)
        except Exception:  # noqa: BLE001 — a full queue must not fail the run
            logger.debug("Dropping assist event for a stalled subscriber")
    return event


async def subscribe(session_id: str):
    """Replay this session's events, then tail live ones. Ends when the session does."""
    session = get_session(session_id)
    if session is None:
        return
    queue: asyncio.Queue = asyncio.Queue()
    # Registered before the history is snapshotted, with no await between the two, so no
    # event can be missed or delivered twice.
    session._queues.append(queue)
    history = list(session.events)
    try:
        for event in history:
            yield event
            if event["type"] == "assist_status" and event.get("status") in _TERMINAL:
                return
        while True:
            event = await queue.get()
            yield event
            if event["type"] == "assist_status" and event.get("status") in _TERMINAL:
                return
    finally:
        if queue in session._queues:
            session._queues.remove(queue)


_TERMINAL = ("proposed", "failed", "applied", "rejected")


# --------------------------------------------------------------------------- the tools


def _tool_schemas() -> List[Dict[str, Any]]:
    """The read-only tools, as OpenAI-compatible function definitions.

    Six read the pack and one reads the method library. All seven read: the guarantee that
    nothing reaches disk until a human approves the plan is a fact about this list, so a tool
    added here must be incapable of writing.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "pack_summary",
                "description": (
                    "The pack's own vocabulary: entity types, source names, ruleset keys, "
                    "shared-check ids, and the condition kinds the engine can evaluate. "
                    "Call this FIRST — it is what stops you inventing a condition kind or "
                    "referring to a source that does not exist."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Every file in the pack, with its size and line count.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "glob": {
                            "type": "string",
                            "description": "optional substring or glob, e.g. shared/checks/*",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "One file's text with 1-indexed line numbers. Pass start_line/end_line "
                    "to read part of a large file. Line numbers from here are what an edit "
                    "op's start_line/end_line must use."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Regular-expression search across the pack. Returns path, line number "
                    "and the matching line."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "glob": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate",
                "description": (
                    "Run the pack checker and return its findings. Use it to see what is "
                    "already broken before proposing a change, and to understand what the "
                    "checker will say about yours."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dry_run",
                "description": (
                    "Evaluate the pack's rulesets against the rows real past runs actually "
                    "retrieved, and report pass/fail/unknown per condition. This is the only "
                    "tool that can see the failure the checker cannot: a condition that is "
                    "valid YAML, names a real leaf, and reads `unknown` on every row that has "
                    "ever come back — it never votes and the report never says so. Read the "
                    "`ALWAYS UNKNOWN` marks and the closing 'what this reading cannot see' "
                    "list, which names the direction each bound errs in."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ruleset": {
                            "type": "string",
                            "description": (
                                "One ruleset key, from pack_summary. Name it when you are "
                                "working on one procedure: the run budget is shared, so "
                                "asking for all of them spends it round-robin."
                            ),
                        }
                    },
                },
            },
        },
        pack_skills.tool_schema(),
    ]


def pack_summary(pack: str) -> Dict[str, Any]:
    """The pack's declared names, plus the engine's own condition vocabulary.

    Read straight off the YAML rather than through the pack loader, for the reason the
    checker does the same: the loader swallows a parse error and returns an empty document,
    so a summary built on it would describe a broken pack as an empty one, and an
    assistant told a pack has no sources will happily propose adding the ones already
    there.
    """
    root = pack_store.pack_dir(pack)
    out: Dict[str, Any] = {
        "pack": pack_store.safe_pack_name(pack),
        "condition_kinds": sorted(pack_validate.condition_kinds()),
        "expected_label_kinds": sorted(pack_validate.expected_label_kinds()),
        "editable_suffixes": sorted(pack_store.EDITABLE_SUFFIXES),
    }

    def _load(rel: str) -> Any:
        path = root / rel
        if not path.is_file():
            return None
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except yaml.YAMLError as e:
            out.setdefault("unreadable", []).append(f"{rel}: {e.__class__.__name__}")
            return None

    glossary = _load("entity_glossary.yaml") or {}
    entities = glossary.get("entities") if isinstance(glossary, dict) else None
    out["entity_types"] = sorted(
        {
            str(e.get("type"))
            for e in (entities or [])
            if isinstance(e, dict) and e.get("type")
        }
    )
    catalog = _load("source_catalog.yaml") or {}
    sources = catalog.get("sources") if isinstance(catalog, dict) else None
    out["source_names"] = sorted(
        {
            str(s.get("name"))
            for s in (sources or [])
            if isinstance(s, dict) and s.get("name")
        }
    )
    rulesets: List[str] = []
    conditions_by_ruleset: Dict[str, int] = {}
    for path in sorted(root.rglob("rule*.yaml")):
        data = _load(str(path.relative_to(root)))
        block = (data or {}).get("verdicts") if isinstance(data, dict) else None
        if not isinstance(block, dict):
            continue
        for key, spec in block.items():
            rulesets.append(f"{path.relative_to(root)}:{key}")
            if isinstance(spec, dict):
                conditions_by_ruleset[str(key)] = len(spec.get("conditions") or [])
    out["rulesets"] = rulesets
    out["conditions_per_ruleset"] = conditions_by_ruleset
    library: Dict[str, Any] = {}
    checks_dir = root / "shared" / "checks"
    if checks_dir.is_dir():
        for path in sorted(checks_dir.glob("*.yaml")):
            data = _load(str(path.relative_to(root)))
            if isinstance(data, dict):
                for cid in data:
                    library[f"{path.stem}/{cid}"] = True
    out["shared_check_ids"] = sorted(library)
    out["use_cases"] = (
        sorted(p.name for p in (root / "use_cases").iterdir() if p.is_dir())
        if (root / "use_cases").is_dir()
        else []
    )
    return out


def _numbered(text: str, first_line: int) -> str:
    lines = text.splitlines()
    width = len(str(first_line + len(lines) - 1))
    return "\n".join(
        f"{str(first_line + i).rjust(width)}\t{line}" for i, line in enumerate(lines)
    )


def _tool_read_file(pack: str, args: Dict[str, Any]) -> str:
    rel = str(args.get("path") or "")
    info = pack_store.read_file(pack, rel)
    lines = info["text"].splitlines()
    total = len(lines)
    start = max(1, int(args.get("start_line") or 1))
    end = int(args.get("end_line") or total)
    end = min(max(end, start), total)
    chunk = "\n".join(lines[start - 1 : end])
    header = f"{info['path']} ({info['kind']}, {total} lines, {info['bytes']} bytes)"
    if len(chunk) > READ_CHARS_PER_CALL:
        # Marked, never silent: a model that cannot see it got a fragment will propose an
        # edit against the fragment, and the line numbers in that proposal will be right
        # for text it never read.
        kept = chunk[:READ_CHARS_PER_CALL]
        shown = kept.count("\n") + start
        return (
            f"{header}\nlines {start}-{end}, TRUNCATED at line {shown} of {total} "
            f"({READ_CHARS_PER_CALL} characters) — read a narrower range for the rest\n"
            + _numbered(kept, start)
        )
    return f"{header}\nlines {start}-{end}\n" + _numbered(chunk, start)


def _matches_glob(rel: str, pattern: str) -> bool:
    """Substring or glob, because a model writes both and neither is wrong."""
    if not pattern:
        return True
    from fnmatch import fnmatch

    return pattern in rel or fnmatch(rel, pattern) or fnmatch(rel, pattern + "*")


def _tool_list_files(pack: str, args: Dict[str, Any]) -> str:
    glob = str(args.get("glob") or "")
    tree = pack_store.pack_tree(pack)
    rows = [
        f"{n['path']}\t{n['kind']}\t{n['lines']} lines\t{n['bytes']} bytes"
        + ("" if n["editable"] else "\t[too large to patch whole — read a range]")
        for n in tree["nodes"]
        if not n["dir"] and _matches_glob(n["path"], glob)
    ]
    if not rows:
        return f"no files match {glob!r}" if glob else "this pack has no files"
    clipped = rows[:MAX_LIST_FILES]
    note = (
        f"\n[{len(rows) - len(clipped)} more not shown — narrow the glob]"
        if len(rows) > len(clipped)
        else ""
    )
    return "\n".join(clipped) + note


def _tool_search(pack: str, args: Dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "")
    if not pattern:
        return "search needs a pattern"
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        return f"that is not a valid regular expression: {e}"
    glob = str(args.get("glob") or "")
    root = pack_store.pack_dir(pack)
    hits: List[str] = []
    truncated = False
    for node in pack_store.pack_tree(pack)["nodes"]:
        if node["dir"] or not node.get("text") or not _matches_glob(node["path"], glob):
            continue
        try:
            text = (root / node["path"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{node['path']}:{i}\t{line.strip()[:200]}")
                if len(hits) >= MAX_SEARCH_HITS:
                    truncated = True
                    break
        if truncated:
            break
    if not hits:
        return f"no match for {pattern!r}"
    return "\n".join(hits) + (
        f"\n[stopped at {MAX_SEARCH_HITS} hits — narrow the pattern]"
        if truncated
        else ""
    )


def _tool_validate(pack: str, args: Dict[str, Any]) -> str:
    result = pack_validate.validate_pack(pack_store.pack_dir(pack))
    head = (
        f"{result['errors']} error(s), {result['warnings']} warning(s), "
        f"{result['infos']} note(s)"
    )
    rows = [
        f"{d['severity']}\t{d['code']}\t{d.get('path') or 'pack'}"
        + (f":{d['line']}" if d.get("line") else "")
        + f"\t{d['message']}"
        for d in result.get("diagnostics", [])
    ]
    return head + ("\n" + "\n".join(rows) if rows else "")


def _tool_dry_run(pack: str, args: Dict[str, Any]) -> str:
    """Replay the pack's rulesets over stored evidence.

    The tool budget rather than the preview one, and the ruleset argument matters more here
    than a bigger budget would: a model asking about one procedure gets its runs spent on
    that procedure's own conditions instead of round-robined across nine.
    """
    raw = args.get("ruleset") or args.get("rulesets") or None
    keys = [raw] if isinstance(raw, str) else list(raw) if raw else None
    report = pack_dry_run.dry_run(
        pack_store.pack_dir(pack),
        ruleset_keys=[str(k) for k in keys if str(k).strip()] if keys else None,
        max_runs=pack_dry_run.TOOL_MAX_RUNS,
        max_seconds=pack_dry_run.TOOL_MAX_SECONDS,
    )
    return pack_dry_run.render(report)


_TOOLS = {
    "pack_summary": lambda pack, args: json.dumps(pack_summary(pack), indent=1),
    "list_files": _tool_list_files,
    "read_file": _tool_read_file,
    "search": _tool_search,
    "validate": _tool_validate,
    "dry_run": _tool_dry_run,
    # Takes no pack: a skill is method, not domain, and is identical for every pack installed.
    "read_skill": lambda pack, args: pack_skills.read(str(args.get("name") or "")),
}


def dispatch_tool(pack: str, name: str, args: Dict[str, Any]) -> str:
    """Run one tool. Every failure comes back as text, never as an exception.

    A raise here would end the loop on a mistyped path, but a wrong path is ordinary
    model behaviour and the correct response is to tell it so, in the tool result, where it
    can act on it. The same applies to an unknown tool name: the model chose from a list it
    was given, so a name outside that list is a fault worth reporting back rather than a
    reason to abandon a session that may already have found what it needed.
    """
    fn = _TOOLS.get(str(name))
    if fn is None:
        return f"there is no tool called {name!r}. The tools are: " + ", ".join(
            sorted(_TOOLS)
        )
    try:
        return str(fn(pack, args or {}))
    except pack_store.PackStoreError as e:
        return f"{name} refused: {e}"
    except Exception as e:  # noqa: BLE001 — a tool fault is a result, not a crash
        logger.warning("Assist tool %s failed on %s: %s", name, pack, e)
        return f"{name} failed: {e}"


# -------------------------------------------------------------------------- the prompts

SYSTEM_PROMPT = """\
You are editing an AFIR domain knowledge pack. The pack is the ONLY place this system \
holds domain knowledge — the engine itself is generic — so a pack file is not \
configuration, it is the investigation's subject-matter expertise written down.

The layout you are working in:

  entity_glossary.yaml      the entity types incidents are described with, their surface
                            forms, recognition hints and field aliases
  source_catalog.yaml       every retrievable source: what it holds, what it CANNOT
                            answer, when it is worth asking, how entities bind to its
                            columns. Uses YAML anchors (&name / *name).
  domain_vocabulary.yaml    this domain's own words. A guard test asserts the engine's
                            own source code never uses them.
  shared/checks/*.yaml      a library of reusable check MECHANICS, imported by a
                            condition as `use: <file stem>/<check id>`
  shared/concepts/*.md      notes about the DATA, readable by every use case
  use_cases/<name>/rules.yaml       the rulesets: conditions, their weighting, the verdict
  use_cases/<name>/reporting.yaml   the report's own wording for that use case
  use_cases/<name>/playbooks/*.md   the procedure
  schemas/*.yaml            generated inventories of a source's columns. Large. Read a
                            line range; never rewrite one whole.

Seven rules that come from how the engine actually reads these files:

1. A shared check holds MECHANICS only (its kind, field paths, a number the data
   dictates). The WEIGHTING belongs to whichever ruleset imports it (decisive, polarity,
   report_group, order). An override is a ONE-LEVEL merge and a list REPLACES.
2. A condition's `label` states its REQUIREMENT. For an exclusion a failure negates the
   label; for a fraud indicator a failure AFFIRMS it and the report prints the label as
   the finding. So importing a check across a polarity means overriding `label` so that
   the failing side reads true. Nothing enforces this and the engine cannot negate prose.
3. Only the condition kinds `pack_summary` lists are evaluated. Any other `kind` falls
   through and the condition is never evaluated at all — which looks exactly like a source
   that returned nothing.
4. An unknown key is SILENTLY DROPPED by the model layer. A key that is not read by the
   engine does nothing at all, and nothing reports it.
5. A file that stops parsing does not raise: the loader hands the engine an EMPTY
   document. A pack with a broken catalogue loads with zero sources and every check
   resolves to unknown, which reads in the final report as the data having nothing to say.
6. `all_of` / `any_of` / `none_of` compose child conditions and produce ONE report line, so
   the parent owns every weighting key (decisive, polarity, report_group, label,
   fail_detail) and a child declaring one is inert. Their logic is three-valued and
   `unknown` never reads as `pass`: `all_of` needs every child to pass, `any_of` any child,
   `none_of` every child to fail, and a remaining unknown that could change the answer makes
   the parent unknown.
7. A bound you do not declare is not a small bound — the engine returns `unknown`, so a
   decisive condition with no bound is inert. And a truncated source is a BOUND on the value
   rather than the value, so a comparison only stands where more rows could not overturn it:
   a count/sum/max exceeding its bound stands, a min below it stands, and an average or a
   ratio is always unknown under truncation.

Explore before you propose. Call `pack_summary` first, then read the files you intend to
change — you need real line numbers and the real surrounding text. Prefer the smallest
edit that does the job: these files are heavily commented and use YAML anchors, and a
whole-file rewrite destroys both. When the change is to a condition, `dry_run` is the one
tool that can tell you whether it will ever answer anything: the checker proves the engine
reads a declaration, not that any row has ever satisfied it.
"""


def system_prompt() -> str:
    """The system turn: the layout and rules above, plus the method library's index.

    The index is built rather than written into the literal, so installing a skill needs no code
    change and a removed one cannot leave a dangling reference. Only the index (one line per
    skill) is unconditional; the bodies are selected per question by ``pack_skills.select`` and
    ride in the user turn, because the whole library injected every time would cost more context
    than the pack files it exists to help read.
    """
    index = pack_skills.index_text()
    if not index:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + "\nThis project keeps its pack-authoring METHOD written down, and the documents "
        "relevant to what you were asked are included in the request below. The full set:\n\n"
        + index
        + "\n\nUse `read_skill` for one that is listed but not included, if the work turns out "
        "to touch it.\n"
    )


PLAN_PROMPT = """\
Now produce the edit plan.

Rules for the ops, all of them consequences of how the plan is applied:

* `patch` replaces the inclusive line range [start_line, end_line] of an EXISTING file.
  Use the line numbers exactly as `read_file` showed them. `text` is the replacement for
  that whole range, with the same indentation the surrounding file uses.
* `expect_first_line` and `expect_last_line` must be the CURRENT text of the lines at
  start_line and end_line, copied verbatim. They are checked before anything is written:
  if the file moved since you read it the whole plan is refused rather than writing to the
  wrong place. A patch without them is refused.
* `create` writes a new file; put the whole content in `text` and leave the line numbers
  at 0.
* `delete` removes a file. It is refused unless the operator separately allows deletions,
  so use it only when removal is genuinely the change being asked for.
* Every path is relative to the pack root, and only .yaml, .yml, .md and .txt can be
  written.
* Keep the surrounding comments. If a comment above your change would become wrong,
  patch it in the same op — a comment that contradicts the code below it is worse than
  no comment.

Say what you could not determine in `questions` instead of guessing. A guess inside an
approved diff is indistinguishable from a finding, and the operator cannot tell them
apart from the diff alone.

Where a method skill in this request states a rule about what you are changing, the plan
complies with it or `notes` says why it does not. In particular: anything a skill says must
be MEASURED against the live source — a binding, a threshold, a filter guarantee, a
cardinality — is a `questions` entry naming the measurement, never a value you chose. A
plausible number in an approved diff is the one failure none of these files can catch.
"""


def _budget_prompt(reason: str) -> str:
    return (
        f"Your exploration budget is spent ({reason}). Do not call any more tools. "
        "Produce the best plan you can from what you have already read, and list what you "
        "were unable to check in `questions` — a plan built on partial exploration is "
        "acceptable, one that hides that it was is not."
    )


# --------------------------------------------------------------------------- the loop


def _assistant_turn(message: Any) -> Dict[str, Any]:
    """The assistant message, converted back to the dict shape the API expects.

    ``tool_call`` hands back the SDK's own object; appending it verbatim works on some
    clients and not others, and the tests hand in fakes. Rebuilding it explicitly means
    the loop depends only on the documented wire shape.
    """
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        calls.append(
            {
                "id": str(getattr(call, "id", "") or ""),
                "type": "function",
                "function": {
                    "name": str(getattr(fn, "name", "") or ""),
                    "arguments": str(getattr(fn, "arguments", "") or "{}"),
                },
            }
        )
    turn: Dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
    }
    if calls:
        turn["tool_calls"] = calls
    return turn


def _parse_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class PackAssistant:
    """Runs the explore→propose loop for one pack. Holds no state between sessions."""

    def __init__(
        self,
        llm_client,
        *,
        max_turns: int = MAX_TURNS,
        max_tool_bytes: int = MAX_TOOL_BYTES,
    ):
        self.llm = llm_client
        self.max_turns = max(1, int(max_turns))
        self.max_tool_bytes = max(1, int(max_tool_bytes))

    async def run(
        self,
        session: AssistSession,
        *,
        focus: Optional[List[str]] = None,
        guidance: str = "",
        prior_plan: Optional[Dict[str, Any]] = None,
        allow_delete: bool = False,
    ) -> AssistSession:
        """Explore, then propose. Never raises; a failure is reported on the session.

        The caller is an HTTP handler that has already answered, so an exception here would
        surface only in the log while the UI waited on a session that never finished.
        """
        session.status = "exploring"
        session.focus = [str(p) for p in (focus or [])]
        emit(session, "assist_status", status="exploring", message="reading the pack")
        # After the status, never before: the first event a subscriber replays is the session's
        # own state transition, and a note arriving ahead of it is a note about a session the UI
        # does not yet know has started.
        self._attach_skills(session, guidance)
        try:
            messages = await self._opening_with_images(session, guidance, prior_plan)
            messages = await self._explore(session, messages)
            plan = await self._propose(session, messages)
            session.plan = plan
            session.status = "checking"
            emit(
                session,
                "assist_status",
                status="checking",
                message=f"{len(plan.ops)} proposed edit(s) — checking the after-state",
                ops=len(plan.ops),
            )
            # On a thread, and the status above exists because of it: the validation and the
            # dry run are tens of seconds of pure CPU, and this coroutine shares its loop with
            # every running job's SSE stream.
            checks = await asyncio.to_thread(
                plan_checks, session.pack, plan, allow_delete=allow_delete
            )
            session.preview = plan_preview(
                session.pack, plan, allow_delete=allow_delete, checks=checks
            )
            session.status = "proposed"
            emit(
                session,
                "assist_status",
                status="proposed",
                message=f"{len(plan.ops)} proposed edit(s)",
                ops=len(plan.ops),
                blocked=len(session.preview.get("errors") or []),
            )
        except Exception as e:  # noqa: BLE001 — the session IS the error channel
            logger.error("Assist session %s failed: %s", session.id, e)
            session.status = "failed"
            session.error = f"{e.__class__.__name__}: {e}"
            emit(session, "assist_status", status="failed", message=session.error)
        return session

    # -- the method library ------------------------------------------------

    def _attach_skills(self, session, guidance: str = "") -> None:
        """Pick the method documents this request needs, and say which they are.

        Selected here and not offered as a suggestion, because an opt-in library is one a model
        can decline invisibly: a plan written without the probe-before-declare rules comes back
        looking exactly like one written with them. The same reasoning that keeps every query
        guarantee out of the retrieval prompts and in a post-generation rewrite.

        The operator's own words are matched, plus the focus paths and any rejection guidance;
        a correction is usually where the real subject of the change first gets named.
        """
        matched = pack_skills.select(
            " ".join([session.question, str(guidance or "")]), focus=session.focus
        )
        session.skill_text = pack_skills.injected_text(matched)
        session.skills = [
            {
                "name": skill.name,
                "why": "always" if skill.always else ", ".join(sorted(triggers)[:6]),
                "chars": skill.chars,
            }
            for skill, triggers in matched
        ]
        if session.skills:
            emit(
                session,
                "assist_note",
                message="method skills applied: "
                + ", ".join(s["name"] for s in session.skills),
                skills=list(session.skills),
            )
        for problem in pack_skills.problems():
            # A skill that failed to parse is a capability this session silently does not have.
            logger.warning("Skill library problem: %s", problem)
            emit(session, "assist_note", message=f"skill not loaded — {problem}")

    # -- the opening context ----------------------------------------------

    async def _opening_with_images(self, session, guidance, prior_plan):
        """The opening turn, with images if this endpoint takes them; probed, not assumed.

        Image support is not a capability flag: it depends on whatever model is deployed
        behind ``base_url``. The probe is its own call rather than turn-0 of
        ``_explore``, because a turn-0 failure is diagnosed as "no tool calling" and a
        rejected image would be reported as the wrong thing. Either way ``image_mode``
        is set to a fact the UI prints: ``read``, ``text_only``, or ``none``.
        """
        images = [a for a in session.attachments if a.get("kind") == "image"]
        if not images:
            session.image_mode = "none"
            return self._opening_messages(session, session.focus, guidance, prior_plan)

        with_images = self._opening_messages(
            session, session.focus, guidance, prior_plan, images=True
        )
        try:
            # One token of output is enough: the refusal, when it comes, is on the way IN.
            await self.llm.complete(
                [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": "Reply with the word ok."}]
                            + pack_attachments.image_blocks(images[:1])
                        ),
                    }
                ]
            )
        except Exception as e:  # noqa: BLE001 — any refusal shape counts as a refusal
            logger.warning("Endpoint refused image content (%s); using placeholders", e)
            session.image_mode = "text_only"
            emit(
                session,
                "assist_note",
                message=(
                    f"this endpoint cannot read images — {len(images)} attached "
                    "image(s) were NOT seen; describe them in the question instead"
                ),
            )
            return self._opening_messages(
                session, session.focus, guidance, prior_plan, images=False
            )
        session.image_mode = "read"
        emit(
            session,
            "assist_note",
            message=f"{len(images)} image(s) sent to the model",
        )
        return with_images

    def _opening_messages(self, session, focus, guidance, prior_plan, *, images=True):
        """The first user turn. ``images=False`` rebuilds it for an endpoint that refused them.

        The image half is a *content block list* rather than more text, which is why this is
        rebuildable at all: whether the deployed endpoint accepts blocks is not knowable in
        advance, so the caller probes with ``images=True`` and, on a refusal, calls again
        with ``False`` to get the same question with every image replaced by a placeholder
        naming it. Both forms carry the same words; only the images differ.
        """
        parts = [f"Pack: {session.pack}\n\nWhat is being asked:\n{session.question}"]
        if session.skill_text:
            # Ahead of the question's own detail: these are the rules the reading that follows
            # is meant to be done under, and they are also what survives into the no-tool
            # fallback, which appends its bundle to this same turn.
            parts.append(session.skill_text)
        if focus:
            parts.append(
                "The operator pointed at these files as the place to start (you may read "
                "others):\n" + "\n".join(f"- {p}" for p in focus)
            )
        for att in session.attachments:
            note = att.get("note") or ""
            if att.get("kind") == "image":
                # Named in the text either way. With images working, this labels the block
                # that follows so the model can tell two diagrams apart; without, the
                # placeholder is the only trace the operator attached anything.
                parts.append(
                    f"Attached image — {att.get('name')}"
                    if images
                    else pack_attachments.image_placeholder(att)
                )
                continue
            parts.append(
                f"Attached — {att.get('name')} ({att.get('kind')}):\n"
                + str(att.get("text") or "")
                + (f"\n[{note}]" if note else "")
            )
        if prior_plan:
            # The rejected plan is quoted rather than summarised, because the operator's
            # correction is usually about a specific op and a paraphrase loses which one.
            parts.append(
                "A previous proposal was REJECTED. It was:\n"
                + json.dumps(prior_plan, indent=1)[:8000]
            )
        if guidance:
            parts.append(
                "The operator's correction, which is the reason for this second attempt:\n"
                + guidance
            )
        text = "\n\n".join(parts)
        blocks = pack_attachments.image_blocks(session.attachments) if images else []
        # A plain string when there are no images, so the overwhelming majority of requests
        # keep the exact shape every endpoint in this repo already handles. The block list is
        # only built when there is actually an image to carry.
        content = ([{"type": "text", "text": text}] + blocks) if blocks else text
        return [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": content},
        ]

    # -- exploration -------------------------------------------------------

    async def _explore(self, session, messages):
        """Run tool rounds until the model stops asking, or a budget runs out."""
        tools = _tool_schemas()
        for turn in range(self.max_turns):
            try:
                message = await self.llm.tool_call(
                    messages, tools, stage="pack_assistant"
                )
            except Exception as e:  # noqa: BLE001
                if turn == 0:
                    # The endpoint does not support tool calling. Fall back to a
                    # pre-loaded bundle and report the mode so the operator can tell.
                    logger.warning(
                        "Tool calling unavailable (%s); falling back to a single shot",
                        e,
                    )
                    session.tool_mode = "single_shot"
                    emit(
                        session,
                        "assist_note",
                        message=(
                            "this endpoint cannot call tools — proposing from a "
                            "pre-loaded bundle instead of exploring"
                        ),
                    )
                    return self._fallback_messages(session, messages)
                raise
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                if turn == 0:
                    # Offered tools and used none. Not an error, but it means the plan
                    # will be written from the question alone, so the bundle goes in.
                    session.tool_mode = "single_shot"
                    emit(
                        session,
                        "assist_note",
                        message="the model asked for nothing — pre-loading the pack instead",
                    )
                    return self._fallback_messages(session, messages)
                return messages
            session.turns = turn + 1
            messages = messages + [_assistant_turn(message)]
            for call in calls:
                fn = getattr(call, "function", None)
                name = str(getattr(fn, "name", "") or "")
                args = _parse_args(getattr(fn, "arguments", "{}"))
                result = dispatch_tool(session.pack, name, args)
                session.tool_bytes += len(result.encode("utf-8"))
                session.trail.append(
                    {
                        "turn": session.turns,
                        "tool": name,
                        "args": args,
                        "bytes": len(result.encode("utf-8")),
                        "preview": result[:400],
                    }
                )
                emit(
                    session,
                    "assist_tool",
                    turn=session.turns,
                    tool=name,
                    args=args,
                    bytes=len(result.encode("utf-8")),
                    preview=result[:400],
                )
                messages = messages + [
                    {
                        "role": "tool",
                        "tool_call_id": str(getattr(call, "id", "") or ""),
                        "content": result,
                    }
                ]
            if session.tool_bytes >= self.max_tool_bytes:
                session.budget_spent = f"read {session.tool_bytes} bytes"
                emit(session, "assist_note", message="tool-output budget spent")
                return messages + [
                    {"role": "user", "content": _budget_prompt(session.budget_spent)}
                ]
        session.budget_spent = f"used all {self.max_turns} exploration turns"
        emit(session, "assist_note", message="turn budget spent")
        return messages + [
            {"role": "user", "content": _budget_prompt(session.budget_spent)}
        ]

    def _fallback_messages(self, session, messages):
        """Pre-load a bounded bundle for an endpoint that cannot explore.

        Deliberately the checker's findings and the tree before any file content: they are
        what say whether the pack is coherent, and they cost a few hundred bytes against a
        schema file's several hundred thousand.
        """
        bundle = [
            "You cannot call tools, so the pack is summarised for you below.",
            "SUMMARY:\n" + json.dumps(pack_summary(session.pack), indent=1),
            "CHECKER:\n" + _tool_validate(session.pack, {}),
            "FILES:\n" + _tool_list_files(session.pack, {}),
        ]
        used = sum(len(b) for b in bundle)
        for rel in self._fallback_files(session):
            if used >= FALLBACK_TOTAL_CHARS:
                bundle.append(f"[not included, budget spent: {rel}]")
                continue
            try:
                text = _tool_read_file(session.pack, {"path": rel})[
                    :FALLBACK_FILE_CHARS
                ]
            except pack_store.PackStoreError as e:
                bundle.append(f"[could not read {rel}: {e}]")
                continue
            used += len(text)
            bundle.append("FILE:\n" + text)
        session.tool_bytes = used
        return messages + [{"role": "user", "content": "\n\n".join(bundle)}]

    def _fallback_files(self, session) -> List[str]:
        """Which files to pre-load: the operator's focus, else the pack's core."""
        chosen = list(session.focus)
        if chosen:
            return chosen[:6]
        tree = pack_store.pack_tree(session.pack)
        core = [
            n["path"]
            for n in tree["nodes"]
            if not n["dir"]
            and n["editable"]
            and n["kind"] in ("glossary", "catalog", "ruleset", "shared_check")
        ]
        return core[:4]

    # -- the plan ----------------------------------------------------------

    async def _propose(self, session, messages) -> EditPlan:
        session.status = "proposing"
        emit(session, "assist_status", status="proposing", message="writing the plan")
        plan = await self.llm.structured_output(
            messages + [{"role": "user", "content": PLAN_PROMPT}],
            EditPlan,
            stage="pack_assistant",
        )
        # Duck-typed rather than isinstance-checked: this module is reachable under both
        # import styles, so the same class has two identities and a fake client in a test
        # hands back its own.
        if not hasattr(plan, "ops"):
            raise ValueError("the model did not return an edit plan")
        return plan


# ------------------------------------------------------------------ plan validation


#: What an op may be. Anything else is skipped with a note; see the module docstring.
KNOWN_OPS = ("create", "patch", "delete")


def _op_dict(op: Any) -> Dict[str, Any]:
    if hasattr(op, "model_dump"):
        return op.model_dump()
    return dict(op or {})


def _plan_dict(plan: Any) -> Dict[str, Any]:
    """A plan as a plain dict, whichever way it arrived.

    Normalised once, here, rather than field by field at each use. The obvious form
    (``getattr(plan, "notes", None) or plan.get("notes")``) reads as a fallback but is a
    bug: an EditPlan whose ``notes`` is the default empty list is falsy, so the expression
    falls through to ``.get`` on a model that has no such method. Two shapes reach this
    module (the LLM's model, and a hand-edited plan arriving as JSON from "Edit first") and
    both are first-class, so the conversion belongs at the boundary.
    """
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    return dict(plan or {})


def _prepared_ops(
    pack: str, ops_in: Any, *, allow_delete: bool
) -> Dict[str, Any]:
    """Prepare every op once, in order, collecting failures instead of raising.

    Returns ``{prepared[], views[], errors[], skipped[]}``. ``prepared`` holds only the ops
    that would be written, each with the final text; ``views`` is one entry per op in plan
    order, including a blocked one, because the operator reads the plan as a list.

    Shared by the preview, the checks and the apply on purpose. The ``after`` text here is
    what the candidate tree is validated against AND what the store is handed, so a second
    preparation loop is a second chance for the checked bytes and the written bytes to be
    different bytes.
    """
    out: Dict[str, Any] = {"prepared": [], "views": [], "errors": [], "skipped": []}
    for index, raw in enumerate(ops_in or []):
        d = _op_dict(raw)
        kind = str(d.get("op") or "").strip().lower()
        if kind not in KNOWN_OPS:
            # Skipped, not raised: three good ops must not be lost to one invented verb.
            note = f"op {index + 1}: unknown operation {kind!r} — skipped"
            logger.warning("Assist plan for %s: %s", pack, note)
            out["skipped"].append(note)
            continue
        try:
            prepared = _prepare_op(pack, kind, d, allow_delete=allow_delete)
        except pack_store.PackStoreError as e:
            out["errors"].append(f"op {index + 1} ({d.get('path') or '?'}): {e}")
            out["views"].append(
                {
                    "op": kind,
                    "path": str(d.get("path") or ""),
                    "reason": str(d.get("reason") or ""),
                    "error": str(e),
                    "diff": "",
                }
            )
            continue
        out["prepared"].append(prepared)
        out["views"].append(prepared["view"])
    return out


def plan_preview(
    pack: str,
    plan: Any,
    *,
    allow_delete: bool = False,
    checks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn a plan into what a human approves: per-op diffs, and what blocks it.

    Returns ``{summary, notes, questions, ops[], errors[], skipped[], checks}``. Nothing is
    written. ``errors`` non-empty means :func:`apply_plan` will refuse the whole plan;
    reported per op rather than as one message, because the operator's next move is
    usually to fix one op and re-apply.

    The diff is rendered here, server-side, and the UI never computes one of its own: two
    implementations that disagree produce a diff that is not what gets written, and the
    diff is the entire basis on which the change was approved.

    ``checks`` is a :func:`plan_checks` result computed elsewhere — this function stays cheap
    and synchronous so it can be called from anywhere, while the checks cost tens of seconds
    of CPU and belong on a thread. An error the plan INTRODUCES is folded into ``errors``
    here, so it blocks the apply through the one path that already refuses a plan.
    """
    doc = _plan_dict(plan)
    prep = _prepared_ops(pack, doc.get("ops") or [], allow_delete=allow_delete)
    out: Dict[str, Any] = {
        "summary": str(doc.get("summary") or ""),
        "notes": list(doc.get("notes") or []),
        "questions": list(doc.get("questions") or []),
        "ops": prep["views"],
        "errors": prep["errors"],
        "skipped": prep["skipped"],
        "allow_delete": bool(allow_delete),
        "checks": checks or None,
    }
    out["errors"].extend((checks or {}).get("introduced") or [])
    return out


def _prepare_op(
    pack: str, kind: str, d: Dict[str, Any], *, allow_delete: bool
) -> Dict[str, Any]:
    """Everything an op needs, checked. Raises :class:`PackStoreError` on any problem.

    Deliberately produces the final text in memory and verifies that, rather than trusting
    the store to catch it at write time: the whole plan must be known-good before the first
    byte lands, and the store's own check fires per file.
    """
    relp = pack_store.safe_rel_path(str(d.get("path") or ""))
    rel = str(relp)
    pack_store.require_editable_suffix(relp)
    path = pack_store.pack_dir(pack) / relp
    exists = path.is_file()
    before = path.read_text(encoding="utf-8", errors="replace") if exists else ""

    if kind == "delete":
        if not exists:
            raise pack_store.PackFileNotFound(f"{rel}: no such file to delete")
        if not allow_delete:
            raise pack_store.PackWriteRejected(
                f"{rel}: this plan deletes a file, which needs the deletion box ticked",
                path=rel,
            )
        return {
            "kind": kind,
            "rel": rel,
            "after": None,
            "view": {
                "op": kind,
                "path": rel,
                "reason": str(d.get("reason") or ""),
                "bytes_before": len(before.encode("utf-8")),
                "diff": pack_store.unified_diff(before, "", rel),
            },
        }

    if kind == "create":
        if exists:
            raise pack_store.PackFileExists(
                f"{rel}: already exists — patch it instead of creating it"
            )
        after = str(d.get("text") or "")
    else:
        if not exists:
            raise pack_store.PackFileNotFound(
                f"{rel}: no such file — use op `create` to add it"
            )
        after = _patched_text(rel, before, d)

    size = len(after.encode("utf-8"))
    if size > pack_store.WRITE_MAX_BYTES:
        raise pack_store.PackTooLarge(
            f"{rel}: {size} bytes is over the {pack_store.WRITE_MAX_BYTES}-byte limit"
        )
    # The same verification the store runs, run here first, in memory: the plan has to be
    # known-good in full before the first byte lands, and the store's own check fires one
    # file at a time; by which point earlier files are already written.
    pack_store.verify_candidate(rel, after, pre_image=before if exists else None)
    return {
        "kind": kind,
        "rel": rel,
        "after": after,
        "view": {
            "op": kind,
            "path": rel,
            "reason": str(d.get("reason") or ""),
            "lines": [int(d.get("start_line") or 0), int(d.get("end_line") or 0)],
            "bytes_before": len(before.encode("utf-8")),
            "bytes_after": len(after.encode("utf-8")),
            "diff": pack_store.unified_diff(before, after, rel),
        },
    }


def _patched_text(rel: str, before: str, d: Dict[str, Any]) -> str:
    """Apply one line-range replacement, with the anchors checked first.

    The anchors are mandatory on a patch and that is the point. A line number computed
    against a snapshot is the dangerous kind of stale: it still points at a line, so the
    write succeeds and lands in the wrong place. Comparing the text is what turns that into
    a refusal.
    """
    try:
        start = int(d.get("start_line") or 0)
        end = int(d.get("end_line") or 0)
    except (TypeError, ValueError):
        raise pack_store.PackWriteRejected(
            f"{rel}: start_line and end_line must be numbers", path=rel
        )
    lines = before.splitlines()
    if start < 1 or end < start or end > len(lines):
        raise pack_store.PackWriteRejected(
            f"{rel}: lines {start}-{end} are outside the file (1-{len(lines)})",
            path=rel,
        )
    first = str(d.get("expect_first_line") or "")
    last = str(d.get("expect_last_line") or "")
    if not first or not last:
        raise pack_store.PackWriteRejected(
            f"{rel}: a patch must carry expect_first_line and expect_last_line — without "
            "them a stale line number writes to the wrong place and still succeeds",
            path=rel,
        )
    actual_first, actual_last = lines[start - 1], lines[end - 1]
    if first.strip() != actual_first.strip():
        raise pack_store.PackConflict(
            f"{rel}: line {start} is {actual_first.strip()!r}, not {first.strip()!r} — "
            "the file moved since the plan was made"
        )
    if last.strip() != actual_last.strip():
        raise pack_store.PackConflict(
            f"{rel}: line {end} is {actual_last.strip()!r}, not {last.strip()!r} — "
            "the file moved since the plan was made"
        )
    replacement = str(d.get("text") or "")
    new_lines = lines[: start - 1] + replacement.splitlines() + lines[end:]
    trailing = "\n" if before.endswith("\n") or not before else ""
    return "\n".join(new_lines) + trailing


@contextlib.contextmanager
def _candidate_tree(pack: str, prepared: List[Dict[str, Any]]) -> Iterator[Path]:
    """The pack as the plan would leave it, in a temp directory, for the duration.

    A copy and never the live tree, and not negotiable: everything that reads a pack reads a
    DIRECTORY, so the only way to ask "what would this pack do after the edit" without a
    second implementation of every loader is to build the after-state on disk. Writing it in
    place and reverting would leave a window in which a run picks up an unapproved pack.

    Dot-prefixed entries are excluded, matching what ``validate_pack`` and the loader both
    skip — ``.history/`` is a content-addressed blob store and copying it is the whole cost.
    """
    root = pack_store.pack_dir(pack)
    with tempfile.TemporaryDirectory(prefix="afir-pack-candidate-") as tmp:
        candidate = Path(tmp) / root.name
        shutil.copytree(root, candidate, ignore=shutil.ignore_patterns(".*"))
        for item in prepared:
            target = candidate / item["rel"]
            if item["kind"] == "delete":
                target.unlink(missing_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["after"], encoding="utf-8")
        yield candidate


def _tree_fingerprint(root: Path) -> str:
    """A cheap identity for a pack directory's contents: every file's path, size and mtime."""
    parts: List[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        st = path.stat()
        parts.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


#: ``{pack: (fingerprint, report)}``. Bounded by the number of installed packs.
_BASELINE_CACHE: Dict[str, Tuple[str, Dict[str, Any]]] = {}


def _baseline_validation(pack: str) -> Dict[str, Any]:
    """What the pack reports TODAY, cached against its own tree fingerprint.

    Needed because the gate is on the errors a plan INTRODUCES, which is a difference and
    therefore needs both sides. Cached because ``validate_pack`` costs seconds on a real pack
    and the same before-state is read by the preview and again by the apply; keyed on the
    tree rather than on time, so an edit through any other path invalidates it.
    """
    root = pack_store.pack_dir(pack)
    fingerprint = _tree_fingerprint(root)
    hit = _BASELINE_CACHE.get(pack)
    if hit is not None and hit[0] == fingerprint:
        return hit[1]
    report = pack_validate.validate_pack(root)
    _BASELINE_CACHE[pack] = (fingerprint, report)
    return report


def _error_counts(report: Dict[str, Any]) -> "Counter[Tuple[str, str, str]]":
    """Every error diagnostic as ``(code, path, message)``, with its multiplicity.

    The LINE NUMBER is deliberately excluded from the identity. A patch shifts every line
    below it, so keying on the line would report the file's pre-existing errors as newly
    introduced on any edit near the top — the plan would be refused for a defect it did not
    cause. A multiset rather than a set so two identical messages in one file still count as
    two, and adding a third is visible.
    """
    return Counter(
        (
            str(d.get("code") or ""),
            str(d.get("path") or ""),
            str(d.get("message") or ""),
        )
        for d in (report.get("diagnostics") or [])
        if str(d.get("severity") or "") == "error"
    )


def _format_errors(counts: "Counter[Tuple[str, str, str]]") -> List[str]:
    out: List[str] = []
    for (code, path, message), n in sorted(counts.items()):
        line = f"{path or 'pack'}: {message} ({code})"
        out.extend([line] * max(1, int(n)))
    return out


def _touched_rulesets(prepared: List[Dict[str, Any]]) -> Optional[List[str]]:
    """The ruleset keys this plan could change, or ``None`` for "any of them".

    A ruleset key IS a use-case directory name (``pack.ruleset_key_for``), so a plan confined
    to ``use_cases/<name>/`` can only change that ruleset, and the dry run's run budget is
    better spent entirely on it than round-robined across nine. Anything at the pack root — a
    shared check, the glossary, the catalog — is readable by every use case, so the answer is
    ``None``.

    Errs toward ``None``: a narrowed dry run that missed the ruleset a plan actually changed
    would report that ruleset as unmeasured, which in a preview reads like a clean result.
    """
    keys: List[str] = []
    for item in prepared:
        parts = PurePosixPath(item["rel"]).parts
        if len(parts) >= 2 and parts[0] == "use_cases":
            if parts[1] not in keys:
                keys.append(parts[1])
            continue
        return None
    return keys or None


def plan_checks(
    pack: str,
    plan: Any,
    *,
    allow_delete: bool = False,
    prepared: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
    max_runs: int = pack_dry_run.DEFAULT_MAX_RUNS,
    max_seconds: float = pack_dry_run.DEFAULT_MAX_SECONDS,
) -> Dict[str, Any]:
    """What a plan does to the pack, measured on the after-state rather than described.

    Returns ``{ran, introduced[], resolved[], baseline_errors, candidate_errors,
    candidate_warnings, rulesets, dry_run, problems[], seconds}``. Two questions, one
    candidate tree:

    * would the pack still validate — reported as the errors the plan ADDS, never as the
      candidate's total. A pack with pre-existing errors is the normal state of one being
      worked on, and gating on the total makes the first fix unappliable.
    * would the new conditions ever answer anything — the dry run, which is the only check
      that can see a condition that is valid, resolves, and reads ``unknown`` on every row
      that has ever come back.

    Blocking CPU for tens of seconds, so it is a standalone function with no coroutine in it:
    the caller decides whether that runs on a thread. ``ran`` false with ``problems`` means
    the measurement did not happen — reported, never folded into a clean result, and never
    turned into a refusal either, since a plan is not at fault for a temp directory.
    """
    started = time.monotonic()
    out: Dict[str, Any] = {
        "ran": False,
        "introduced": [],
        "resolved": [],
        "baseline_errors": 0,
        "candidate_errors": 0,
        "candidate_warnings": 0,
        "rulesets": None,
        "dry_run": None,
        "problems": [],
        "seconds": 0.0,
    }
    prep = prepared or _prepared_ops(
        pack, _plan_dict(plan).get("ops") or [], allow_delete=allow_delete
    )
    if prep["errors"]:
        out["problems"].append(
            "not measured: an op could not be prepared, so there is no after-state to "
            "check. Fix the blocked op(s) above and the checks run on the next preview."
        )
        return out
    if not prep["prepared"]:
        out["problems"].append("not measured: the plan changes no file")
        return out
    try:
        with _candidate_tree(pack, prep["prepared"]) as candidate:
            baseline = _baseline_validation(pack)
            report = pack_validate.validate_pack(candidate)
            base_counts, cand_counts = _error_counts(baseline), _error_counts(report)
            out["baseline_errors"] = int(baseline.get("errors") or 0)
            out["candidate_errors"] = int(report.get("errors") or 0)
            out["candidate_warnings"] = int(report.get("warnings") or 0)
            out["introduced"] = _format_errors(cand_counts - base_counts)
            out["resolved"] = _format_errors(base_counts - cand_counts)
            if dry_run:
                keys = _touched_rulesets(prep["prepared"])
                out["rulesets"] = list(keys) if keys else None
                out["dry_run"] = pack_dry_run.as_dict(
                    pack_dry_run.dry_run(
                        candidate,
                        ruleset_keys=keys,
                        max_runs=max_runs,
                        max_seconds=max_seconds,
                    )
                )
            out["ran"] = True
    except Exception as e:  # noqa: BLE001 — an unmeasurable plan is reported, not refused
        logger.warning("Plan checks for %s could not run: %s", pack, e)
        out["problems"].append(
            f"not measured: {type(e).__name__}: {e}. Nothing here says the plan is sound."
        )
    out["seconds"] = round(time.monotonic() - started, 2)
    return out


def apply_plan(
    pack: str,
    plan: Any,
    *,
    allow_delete: bool = False,
    actor: str = "",
    session: str = "",
    checks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write an approved plan. All of it, or none of it.

    Every op is prepared and verified in memory first; a single problem returns without
    touching the pack. Once writing starts, an I/O failure mid-plan restores the files this
    call already changed from the snapshots it just took, and reports ``rolled_back``;
    a partly-applied plan leaves the pack in a state neither the operator's before
    nor their after describes.

    ``checks`` is a :func:`plan_checks` result for THIS plan; absent, it is computed here.
    Either way an error the plan introduces refuses the whole write — a proposal that leaves
    the pack failing validation is one the next run loads as an empty pack, and the operator
    approved a diff rather than that outcome. Passing it in is what lets an HTTP handler pay
    that cost off the event loop; the parameter is not a way to skip the gate, since a plan
    the caller edited after computing them fails to prepare against its own anchors.
    """
    doc = _plan_dict(plan)
    prep = _prepared_ops(pack, doc.get("ops") or [], allow_delete=allow_delete)
    if checks is None:
        # Validation only. The gate is "does this plan break the pack"; the dry run answers
        # "will the new condition ever fire", which is an authoring question for a human
        # reading a preview and gates nothing — running it here would spend a replay budget
        # on every write to produce a number no branch reads.
        checks = plan_checks(
            pack, plan, allow_delete=allow_delete, prepared=prep, dry_run=False
        )
    errors = list(prep["errors"]) + list(checks.get("introduced") or [])
    if errors:
        return {
            "applied": False,
            "error": "The plan was refused; nothing was written",
            "errors": errors,
            "skipped": prep["skipped"],
            "checks": checks,
        }
    prepared = prep["prepared"]

    written: List[Dict[str, Any]] = []
    undo: List[Tuple[str, Optional[str]]] = []
    for item in prepared:
        rel, kind = item["rel"], item["kind"]
        path = pack_store.pack_dir(pack) / rel
        pre = (
            path.read_text(encoding="utf-8", errors="replace")
            if path.is_file()
            else None
        )
        try:
            if kind == "delete":
                result = pack_store.delete_file(pack, rel, actor=actor, session=session)
            elif kind == "create":
                result = pack_store.create_file(
                    pack, rel, item["after"], actor=actor, session=session
                )
            else:
                result = pack_store.write_file(
                    pack,
                    rel,
                    item["after"],
                    actor=actor,
                    session=session,
                    reason="assist",
                )
        except (pack_store.PackStoreError, OSError) as e:
            logger.error("Assist apply failed on %s/%s: %s", pack, rel, e)
            rolled = _roll_back(pack, undo, actor=actor, session=session)
            return {
                "applied": False,
                "error": f"{rel}: {e}",
                "written": written,
                "rolled_back": rolled,
                "checks": checks,
            }
        undo.append((rel, pre))
        written.append(result)
    return {
        "applied": True,
        "written": written,
        "skipped": prep["skipped"],
        "summary": str(doc.get("summary") or ""),
        "checks": checks,
    }


def _roll_back(
    pack: str, undo: List[Tuple[str, Optional[str]]], *, actor: str, session: str
) -> List[str]:
    """Put back what this call changed. Best-effort, and it says what it could not undo."""
    restored: List[str] = []
    for rel, pre in reversed(undo):
        try:
            if pre is None:
                pack_store.delete_file(pack, rel, actor=actor, session=session)
            else:
                path = pack_store.pack_dir(pack) / rel
                if path.is_file():
                    pack_store.write_file(
                        pack, rel, pre, actor=actor, session=session, reason="rollback"
                    )
                else:
                    pack_store.create_file(pack, rel, pre, actor=actor, session=session)
            restored.append(rel)
        except (pack_store.PackStoreError, OSError) as e:
            # Reported rather than swallowed: an un-undone file is the one thing the
            # operator must know about, and history still holds every version.
            logger.error("Could not roll back %s/%s: %s", pack, rel, e)
    return restored
