"""The grammar of one ``projection:`` entry: the name a value arrives under.

A dotted entry is aliased by the generator (dots → underscores); ``AS <alias>`` overrides
that convention and is authoritative. Two readers serve opposite questions:
:func:`projection_names` — what the value is called;
:func:`projection_sources` — what leaves it reads (for schema truncation).
"""

import re
from typing import List, Tuple

#: A trailing ``AS <identifier>``: the alias an entry gives its own result. Matched only at the end
#: of the entry and at paren depth zero outside any quote, or a nested ``CAST(... AS ...)`` reads
#: as the entry's name.
_TRAILING_ALIAS = re.compile(r"\bAS\s+(`?)([A-Za-z_]\w*)\1\s*$", re.IGNORECASE)

#: A dotted identifier run — the shape a field path has inside an expression.
_DOTTED_RUN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")

#: Names bound by a lambda, single (``x -> ...``) or parenthesised (``(k, v) -> ...``). A run
#: rooted at one of these references an ELEMENT of a group rather than a path on the table.
_LAMBDA_PARAMS = re.compile(r"(\([^()]*\)|[A-Za-z_]\w*)\s*->")

#: Anything that makes an entry an expression rather than a bare path. Explicit so a plain path
#: stays on the unchanged route, where the entry is both its own name and its own source.
_EXPRESSION_CHARS = set("()'\"*+-/,|<>= ")


def _strip_literals(text: str) -> str:
    """``text`` with quoted literal contents blanked; length preserved for offset arithmetic in :func:`split_projection`."""
    out = list(text)
    quote = ""
    i = 0
    while i < len(out):
        ch = out[i]
        if quote:
            if ch == quote:
                quote = ""
            else:
                out[i] = " "
        elif ch in "'\"":
            quote = ch
        i += 1
    return "".join(out)


def split_projection(entry: str) -> Tuple[str, str]:
    """``"<expr> AS <alias>"`` → ``(expr, alias)``; else ``(entry, "")``. Alias must be at paren-depth zero (``CAST(... AS ...)`` excluded)."""
    text = (entry or "").strip()
    if not text:
        return "", ""
    masked = _strip_literals(text)
    match = _TRAILING_ALIAS.search(masked)
    if not match:
        return text, ""
    if masked.count("(", 0, match.start()) != masked.count(")", 0, match.start()):
        return text, ""  # the AS is inside a call — a cast's type, not our name
    return text[: match.start()].strip(), match.group(2)


def projection_alias(entry: str) -> str:
    """The alias an entry declares for itself, or ``""`` when it names itself."""
    return split_projection(entry)[1]


def projection_names(entry: str) -> List[str]:
    """The name(s) the row will carry for this entry.

    Aliased entry: the alias. Bare path: the path itself. Unaliased expression: ``[]``
    (returning the text would hand consumers a name matching no column).
    """
    expr, alias = split_projection(entry)
    if alias:
        return [alias]
    if not expr:
        return []
    return [] if set(expr) & _EXPRESSION_CHARS else [expr]


def projection_sources(entry: str) -> List[str]:
    """Field paths this entry reads, in order, de-duplicated. Lambda-param roots excluded."""
    expr, _alias = split_projection(entry)
    if not expr:
        return []
    if not (set(expr) & _EXPRESSION_CHARS):
        return [expr]  # a bare path: unchanged behaviour
    masked = _strip_literals(expr)
    locals_: set = set()
    for raw in _LAMBDA_PARAMS.findall(masked):
        for name in re.findall(r"[A-Za-z_]\w*", raw):
            locals_.add(name)
    out: List[str] = []
    for run in _DOTTED_RUN.findall(masked):
        if run.split(".", 1)[0] in locals_:
            continue
        if run not in out:
            out.append(run)
    return out


def returned_names(sql: str):
    """Column names a SELECT returns, or ``None`` when unknowable (``*``, CTE).

    Counterpart to :func:`projection_names` over a generated statement. ``None`` means
    "no finding", not "no columns".
    """
    text = (sql or "").strip()
    if not text:
        return None
    masked = _strip_literals(text)
    head = re.search(r"\bSELECT\b(.*?)\bFROM\b", masked, re.IGNORECASE | re.DOTALL)
    if not head:
        return None
    if re.match(r"(?is)^\s*WITH\b", masked) or "*" in head.group(1):
        return None
    names = set()
    depth = 0
    item = []
    for ch in head.group(1):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            names.add(_returned_name("".join(item)))
            item = []
        else:
            item.append(ch)
    names.add(_returned_name("".join(item)))
    names |= set(re.findall(r"\bAS\s+`?([A-Za-z_]\w*)`?", masked, re.IGNORECASE))
    return {n for n in names if n}


def _returned_name(item: str) -> str:
    """One SELECT item -> the column name it lands under."""
    expr, alias = split_projection(item)
    if alias:
        return alias
    expr = expr.strip().strip("`")
    if not expr or set(expr) & _EXPRESSION_CHARS:
        return ""  # an unaliased expression: the backend invents the name
    return expr.split(".")[-1]


def colliding_row_names(entries: List[str]) -> "dict[str, list[str]]":
    """Entries that collide on one row column (last writer wins silently, worse than absent).

    Distinct entries only; identical duplicates are redundancy, not collision.
    """
    seen: "dict[str, list[str]]" = {}
    for entry in entries or []:
        if not isinstance(entry, str):
            continue
        name = _returned_name(entry)
        if not name:
            continue
        seen.setdefault(name, []).append(entry.strip())
    return {name: shared for name, shared in seen.items() if len(set(shared)) > 1}


#: A pure member-accessor chain — how an expression's result is reached after the call that
#: produced it (``...).a.b``). Anything else (an index, an operator, a second call) is not a
#: rename of one path, and this module declines to say what it is.
_TRAILING_ACCESS = re.compile(r"^(?:\.[A-Za-z_]\w*)+$")


def projection_renames(entries: List[str]) -> "dict[str, str]":
    """``{alias: the source field path it renames}``, ``""`` where not unambiguously resolvable.

    Needed so a schema check can trace reads through an alias. Resolved only where exactly
    one source path is reachable through member accessors; anything else maps to ``""``.
    """
    out: "dict[str, str]" = {}
    for entry in entries or []:
        if not isinstance(entry, str):
            continue
        expr, alias = split_projection(entry)
        if not alias:
            continue
        sources = projection_sources(expr)
        if len(sources) != 1:
            out[alias] = ""
            continue
        masked = _strip_literals(expr)
        depth = 0
        close = -1
        for index, ch in enumerate(masked):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
                if depth == 0:
                    close = index
        if close < 0:
            out[alias] = sources[0] if expr.strip("`") == sources[0] else ""
            continue
        tail = expr[close + 1 :].strip()
        if not tail:
            out[alias] = sources[0]
        elif _TRAILING_ACCESS.match(tail):
            out[alias] = sources[0] + tail
        else:
            out[alias] = ""
    return out


def expand_projection(entries: List[str]) -> List[str]:
    """Source paths for all entries, flattened; an expression's text matches no column, its source path does."""
    out: List[str] = []
    for entry in entries or []:
        if not isinstance(entry, str):
            continue
        for path in projection_sources(entry):
            if path not in out:
                out.append(path)
    return out
