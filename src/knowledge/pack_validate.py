"""
Diagnostics for one knowledge pack directory.

Severity: error blocks a save (the pack does not work or works while stating something
false; a broken pack reports ``INSUFFICIENT DATA`` rather than a verdict); warning is
reported and never blocks (a declaration is inert or misleading); info surfaces a fact
that is not a defect. ``label-polarity-unaffirmed`` is a one-directional heuristic
because the engine cannot read prose; every other check is mechanical.

Entry point: ``validate_pack(pack_dir)``. Every vocabulary derived here (condition kinds,
link directions, token separators) is read off the engine's own source; when a region
cannot be located its check turns itself off rather than firing everywhere.
"""

import logging
import re
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from src import link_escalation
from src.encoded_fields import decoded_tables
from src.knowledge.pack import (
    EntityDef,
    SourceDef,
    _read_data_dir,
    _read_equivalence_forms,
    _read_shared_checks,
    _resolve_check_imports,
)
from src.utils.paths import REPO_ROOT
from src.utils.projection import (
    colliding_row_names,
    projection_alias,
    projection_names,
    projection_renames,
)

logger = logging.getLogger(__name__)

#: Roots the neutrality collision check scans — the two that must speak no domain at all:
#: the engine itself and the config templates every deployment copies.
_NEUTRALITY_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "config" / "templates")
_NEUTRALITY_SUFFIXES = frozenset(
    {".py", ".yaml", ".yml", ".html", ".js", ".css", ".md"}
)

#: Words the neutrality pattern would flag that the engine legitimately owns — a backend kind
#: names its product, and the engine must spell it. A pack listing one of these is not
#: reporting a leak, so the hit is dropped rather than weakening the pattern for everyone.
_ENGINE_OWNS = frozenset({"servicenow", "kibana", "elk"})

#: Filename a pack uses to declare the vocabulary the engine must never speak.
VOCABULARY_FILE = "domain_vocabulary.yaml"

#: Keys of a ruleset that hold a logical source name. ``_resolve_check_imports`` has already
#: run by the time these are read, so a name introduced by a shared check's mechanics is
#: seen here exactly as the evaluator will see it.
_SOURCE_REF_KEYS = frozenset({"source", "in", "from", "confirm_in"})

#: Cheap caches. The scanned corpora do not change while the process runs, and
#: `validate_pack` is called on every write.
_CORPUS_CACHE: Dict[str, Any] = {}


# ------------------------------------------------------- derived engine facts


def _engine_region(func: str, key: str, checks: str) -> str:
    """The text of one top-level function in ``correlation.py``, cached under ``key``.

    Read off disk rather than imported: ``correlation`` uses flat imports, so importing it
    from here would drag in the pipeline's whole import graph. Unlocatable yields ``""`` and
    a warning; every vocabulary derived from this yields an empty set, which turns its check
    off.
    """
    if key not in _CORPUS_CACHE:
        text = (REPO_ROOT / "src" / "correlation.py").read_text(encoding="utf-8")
        start = text.find(f"\ndef {func}(")
        if start == -1:
            logger.warning(
                "Could not locate %s in correlation.py; %s will be skipped.", func, checks
            )
            _CORPUS_CACHE[key] = ""
            return ""
        rest = text[start + 1 :]
        nxt = re.search(r"\n(?:def |class )", rest)
        _CORPUS_CACHE[key] = rest[: nxt.start()] if nxt else rest
    return _CORPUS_CACHE[key]


def _evaluator_source() -> str:
    """The text of ``correlation._eval_condition``: the region that dispatches on ``kind``."""
    return _engine_region("_eval_condition", "evaluator", "condition-kind checks")


def condition_kinds() -> Set[str]:
    """Every ``kind`` the evaluator dispatches, derived from ``_eval_condition``'s own body.

    Derived rather than listed: a hand-kept list is how a new kind would go unreported
    for months. The region is bounded to this one function to exclude the RAG source
    vocabulary.
    """
    if "kinds" not in _CORPUS_CACHE:
        _CORPUS_CACHE["kinds"] = set(
            re.findall(r'kind == "([a-z_]+)"', _evaluator_source())
        )
    return _CORPUS_CACHE["kinds"]


def link_directions() -> Tuple[str, ...]:
    """The causal axis values an ``entry_signals`` declaration may name, from ``src/links.py``.

    Derived rather than copied: ``links.LINK_DIRECTIONS`` is also what decides a referral's
    window, and ``src/links.py`` imports ``correlation`` at module level, so importing it here
    would fail. Unlocatable yields ``()``, turning the direction check off.
    """
    if "link_directions" not in _CORPUS_CACHE:
        found: Tuple[str, ...] = ()
        path = REPO_ROOT / "src" / "links.py"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        m = re.search(r"\nLINK_DIRECTIONS[^=]*=\s*\(([^)]*)\)", text)
        if m:
            found = tuple(re.findall(r'"([a-z_]+)"', m.group(1)))
        if not found:
            logger.warning(
                "Could not derive LINK_DIRECTIONS from src/links.py; "
                "entry-signal direction checks will be skipped."
            )
        _CORPUS_CACHE["link_directions"] = found
    return _CORPUS_CACHE["link_directions"]


def prose_split_pattern() -> str:
    """The engine's free-text token separator, read from ``correlation._prose_tokens``.

    Derived so the check reports which tokens carry weight under the same splitter the engine
    uses. Unlocatable yields ``""``, turning the check off.
    """
    if "prose_split" not in _CORPUS_CACHE:
        m = re.search(
            r"re\.split\(\s*r\"([^\"]+)\"\s*,\s*text\.lower\(\)\s*\)",
            _engine_region("_prose_tokens", "prose_tokens", "spec-title token checks"),
        )
        if not m:
            logger.warning(
                "Could not derive the prose token separator from correlation._prose_tokens; "
                "spec-title token checks will be skipped."
            )
        _CORPUS_CACHE["prose_split"] = m.group(1) if m else ""
    return _CORPUS_CACHE["prose_split"]


def _spec_tokens(title: str, keys: Any) -> Set[str]:
    """The discriminating vocabulary the scorer builds for one spec: its title plus its keys.

    Mirrors ``_playbook_correlation_spec``'s own construction: prose tokens of the title
    unioned with the prose tokens of every join key, over the engine's derived separator.
    """
    pattern = prose_split_pattern()
    if not pattern:
        return set()

    def toks(text: str) -> Set[str]:
        return {t for t in re.split(pattern, str(text).lower()) if t}

    tokens = toks(title)
    for key in keys if isinstance(keys, list) else []:
        tokens |= toks(key)
    tokens.discard("")
    return tokens


def kinds_reading(key: str) -> Set[str]:
    """The ``kind`` values whose evaluator block mentions ``key``, i.e. that can read it.

    Derived to stay in sync with the evaluator. A key mentioned before the first ``kind ==``
    block belongs to the shared preamble and is attributed to no kind; silence where the
    derivation has no signal, per the rule that an empty vocabulary turns its check off.
    """
    cache_key = f"kinds_reading:{key}"
    if cache_key not in _CORPUS_CACHE:
        body = _evaluator_source()
        blocks = [
            (m.start(), m.group(1)) for m in re.finditer(r'kind == "([a-z_]+)"', body)
        ]
        owners: Set[str] = set()
        for hit in re.finditer(re.escape(key), body):
            owner = ""
            for pos, kind in blocks:
                if pos < hit.start():
                    owner = kind
                else:
                    break
            if owner:
                owners.add(owner)
        _CORPUS_CACHE[cache_key] = owners
    return _CORPUS_CACHE[cache_key]


def expected_label_kinds() -> Set[str]:
    """The ``kind`` values whose evaluator actually reads ``expected_label``.

    Declaring ``expected_label`` on a kind not in this set is a silent no-op. ``distinct_count``
    is the notable case: it renders ``expected`` as the bound ``<= 1`` and never prints
    ``expected_label``, so the pass wording must ride on ``pass_detail`` instead.
    """
    return kinds_reading("expected_label")


def kinds_getting(key: str) -> Set[str]:
    """The ``kind`` values whose evaluator reads ``key`` as a CONDITION key.

    The strict sibling of :func:`kinds_reading`, which matches the bare name anywhere in the
    block and so attributes a key to every kind whose prose or local variables happen to
    contain it — ``form`` is inside ``norm_form`` and inside the word "form". Matching
    ``get("<key>"`` reads only where a value is taken off the condition. Both exist because
    the loose one is right for a key the evaluator reaches through a helper.
    """
    cache_key = f"kinds_getting:{key}"
    if cache_key not in _CORPUS_CACHE:
        body = _evaluator_source()
        blocks = [
            (m.start(), m.group(1)) for m in re.finditer(r'kind == "([a-z_]+)"', body)
        ]
        owners: Set[str] = set()
        for hit in re.finditer(rf'get\("{re.escape(key)}"', body):
            owner = ""
            for pos, kind in blocks:
                if pos < hit.start():
                    owner = kind
                else:
                    break
            if owner:
                owners.add(owner)
        _CORPUS_CACHE[cache_key] = owners
    return _CORPUS_CACHE[cache_key]


def subject_rows_kinds() -> Set[str]:
    """The ``kind`` values whose evaluator reads ``subject_rows``.

    On these kinds ``subject_rows`` is required, not optional: without it every row of the
    source counts as the subject's own and the comparison clears without examining anything.
    """
    return kinds_getting("subject_rows")


def _engine_literals(name: str) -> Tuple[str, ...]:
    """The string members of a module-level tuple constant in ``correlation.py``.

    Derived for the same reason as every other vocabulary here: a hand-kept copy is how a
    widened operator set becomes a spurious error on a pack that is right. Unlocatable
    yields ``()``, which turns the checks built on it off. Composed tuples (``A + B``) are
    read through their own names by the caller.
    """
    cache_key = f"engine_literals:{name}"
    if cache_key not in _CORPUS_CACHE:
        try:
            text = (REPO_ROOT / "src" / "correlation.py").read_text(encoding="utf-8")
        except OSError:
            text = ""
        m = re.search(rf"\n{re.escape(name)}\s*=\s*\(([^)]*)\)", text)
        found = tuple(re.findall(r'"([^"]+)"', m.group(1))) if m else ()
        if not found:
            logger.warning(
                "Could not derive %s from correlation.py; the checks reading it "
                "will be skipped.",
                name,
            )
        _CORPUS_CACHE[cache_key] = found
    return _CORPUS_CACHE[cache_key]


def _engine_mapping_keys(name: str) -> Tuple[str, ...]:
    """The string keys of a module-level dict constant in ``correlation.py``.

    The sibling of ``_engine_literals`` for a vocabulary whose members carry a value as well as
    a name — ``_ORDER_RELATIONS`` maps each relation to its strictness — and which therefore
    cannot be spelled as a tuple. Keys only: a value that happens to be a string is not a
    member. Unlocatable yields ``()`` and turns its checks off, per ``_engine_literals``.
    """
    cache_key = f"engine_mapping_keys:{name}"
    if cache_key not in _CORPUS_CACHE:
        try:
            text = (REPO_ROOT / "src" / "correlation.py").read_text(encoding="utf-8")
        except OSError:
            text = ""
        m = re.search(rf"\n{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", text)
        found = tuple(re.findall(r'"([^"]+)"\s*:', m.group(1))) if m else ()
        if not found:
            logger.warning(
                "Could not derive %s from correlation.py; the checks reading it "
                "will be skipped.",
                name,
            )
        _CORPUS_CACHE[cache_key] = found
    return _CORPUS_CACHE[cache_key]


def compare_operators() -> Tuple[str, ...]:
    """The comparison operators a ``numeric_compare`` may declare.

    ``_COMPARE_OPS`` is composed from two named tuples plus ``"=="``, so it is read through
    its parts; ``==`` is spelled here because it is spelled inline there.
    """
    parts = _engine_literals("_INCREASING_OPS") + _engine_literals("_DECREASING_OPS")
    return (parts + ("==",)) if parts else ()


def form_normalize_kinds() -> Set[str]:
    """The ``kind`` values whose ``normalize:`` names an equivalence FORM.

    ``normalize`` is one key over two vocabularies: these kinds resolve it against
    ``shared/equivalence_forms.yaml``, while the older seams read a fixed mode name. Derived
    from the resolver call rather than listed, because the collision is silent in both
    directions — a form name on a mode seam is ignored, a mode name where a form is expected is
    a form nobody declared.
    """
    if "form_normalize_kinds" not in _CORPUS_CACHE:
        body = _evaluator_source()
        blocks = [
            (m.start(), m.group(1)) for m in re.finditer(r'kind == "([a-z_]+)"', body)
        ]
        owners: Set[str] = set()
        for hit in re.finditer(r'_projection_form\(cond, "normalize"', body):
            owner = ""
            for pos, kind in blocks:
                if pos < hit.start():
                    owner = kind
                else:
                    break
            if owner:
                owners.add(owner)
        _CORPUS_CACHE["form_normalize_kinds"] = owners
    return _CORPUS_CACHE["form_normalize_kinds"]


def form_aggregates() -> Tuple[str, ...]:
    """The aggregates a ``normalize:`` form actually changes.

    ``_FORM_AGGREGATES`` is composed (``("distinct",) + _MODAL_AGGREGATES``), so it is read
    through its parts for the reason ``compare_operators`` is: a regex over the composed
    literal returns the first half and the check would then flag the other half as inert.
    """
    modal = _engine_literals("_MODAL_AGGREGATES")
    return (("distinct",) + modal) if modal else ()


def order_relations() -> Tuple[str, ...]:
    """The ordering relations an ``event_order`` may declare."""
    return _engine_mapping_keys("_ORDER_RELATIONS")


def order_quantifiers() -> Tuple[str, ...]:
    """The quantifiers an ``event_order`` may declare over its timestamp pairs."""
    return _engine_literals("_ORDER_QUANTIFIERS")


def composite_kinds() -> Tuple[str, ...]:
    """The kinds that combine ``children`` rather than reading rows themselves."""
    return _engine_literals("_COMPOSITE_KINDS")


def max_composite_depth() -> int:
    """The engine's composite nesting bound; ``0`` when it cannot be derived (check off)."""
    if "max_composite_depth" not in _CORPUS_CACHE:
        try:
            text = (REPO_ROOT / "src" / "correlation.py").read_text(encoding="utf-8")
        except OSError:
            text = ""
        m = re.search(r"\n_MAX_COMPOSITE_DEPTH\s*=\s*(\d+)", text)
        _CORPUS_CACHE["max_composite_depth"] = int(m.group(1)) if m else 0
    return _CORPUS_CACHE["max_composite_depth"]


def _check_cohort_subject_rows(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Check that a cohort kind declares both ``subject_rows`` and ``subject_scope: false``.

    Both are required together: without ``subject_rows`` the check clears having compared
    nothing; without the widening the cohort's other rows are absent before the comparison.
    Errors on either, and both are reported at once because fixing one without the other is
    worse than fixing neither.
    """
    kinds = subject_rows_kinds()
    if not kinds or kind not in kinds:
        return
    sel = cond.get("subject_rows") or {}
    sel = sel if isinstance(sel, dict) else {}
    if not [
        w
        for w in (sel.get("where") or [])
        if isinstance(w, dict) and str(w.get("field", "") or "").strip()
    ]:
        diags.append(
            _diag(
                "error",
                "cohort-subject-rows-missing",
                f"{cid or '(unnamed)'}: kind {kind!r} declares no `subject_rows.where`, so the "
                "subject's own rows cannot be told from the cohort's others — every row counts "
                "as the subject's own and the check clears having compared nothing",
                path=rel,
                line=line,
                hint=(
                    "declare the clause that locates the subject's rows in this source "
                    "(a `field` bound to the subject's own identity)"
                ),
            )
        )
    if cond.get("subject_scope") is not False:
        diags.append(
            _diag(
                "error",
                "cohort-scope-narrowed",
                f"{cid or '(unnamed)'}: kind {kind!r} without `subject_scope: false` is asked "
                "only about rows already narrowed to the subject, so the cohort's OTHER rows "
                "are gone before the comparison and it can never find one",
                path=rel,
                line=line,
                hint=(
                    "this kind supplies both sides from one source and re-selects the "
                    "subject's own rows itself; the default narrowing removes the half it "
                    "compares against"
                ),
            )
        )


def _row_match_needs_widening() -> bool:
    """Does the rollup apply ``row_match`` only where ``subject_scope`` is ``False``?

    Probed rather than assumed: if the engine changes and this condition stops matching,
    the check turns itself off rather than firing on a version of the engine that lacks the gate.
    """
    if "row_match_gate" not in _CORPUS_CACHE:
        body = _verdict_source()
        _CORPUS_CACHE["row_match_gate"] = bool(
            re.search(r'subject_scope"\)\s+is\s+False', body)
            and re.search(r'get\("row_match"', body)
        )
    return bool(_CORPUS_CACHE["row_match_gate"])


def _check_row_match_scope(
    cond: Dict[str, Any],
    cid: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """``row_match`` with no ``subject_scope: false`` is inert: error.

    Without the widening, ``row_match`` clauses are never applied and the check reads every
    row the retriever's entity filter returned, including rows for other identities.
    """
    clauses = [c for c in (cond.get("row_match") or []) if isinstance(c, dict)]
    if not clauses or not _row_match_needs_widening():
        return
    if cond.get("subject_scope") is not False:
        diags.append(
            _diag(
                "error",
                "row-match-inert",
                f"{cid or '(unnamed)'}: `row_match` is never applied without "
                "`subject_scope: false` — the check keeps the default per-subject rows and the "
                "identity lookup it declares does nothing",
                path=rel,
                line=line,
                hint=(
                    "the two are one declaration: the widening admits the source's other rows "
                    "and `row_match` is what re-scopes them to this identity"
                ),
            )
        )


def pair_by_kinds() -> Set[str]:
    """The ``kind`` values whose evaluator honours ``pair_by``, and the values it accepts.

    Derived from the evaluator's own gate; also populates ``pair_by_values`` as a side effect.
    A ``pair_by`` the evaluator never reaches leaves the comparison pooled across the source
    while the pack reads as though it paired within a record.
    """
    if "pair_by_kinds" not in _CORPUS_CACHE:
        body = _evaluator_source()
        gate = re.search(r"kind in \(([^)]*)\)[^\n]*\n[^\n]*pair_by", body)
        _CORPUS_CACHE["pair_by_kinds"] = (
            set(re.findall(r'"([a-z_]+)"', gate.group(1))) if gate else set()
        )
        _CORPUS_CACHE["pair_by_values"] = set(
            re.findall(r'get\("pair_by".*?== "([a-z_]+)"', body)
        )
    return _CORPUS_CACHE["pair_by_kinds"]


def pair_by_values() -> Set[str]:
    """The ``pair_by`` values the evaluator compares against (see ``pair_by_kinds``)."""
    if "pair_by_values" not in _CORPUS_CACHE:
        pair_by_kinds()
    return _CORPUS_CACHE["pair_by_values"]


def _verdict_source() -> str:
    """The text of ``correlation.evaluate_verdict``, read off disk (see ``_evaluator_source``).

    A separate region from ``_evaluator_source`` because the condition's ``kind`` is dispatched
    by ``_eval_condition`` while row scoping is decided by the rollup in ``evaluate_verdict``.
    Scanning one for the other's keys silently derives empty sets, turning those checks off.
    """
    return _engine_region("evaluate_verdict", "verdict", "subject-scope checks")


def subject_scope_values() -> Set[str]:
    """The non-boolean ``subject_scope`` values the rollup compares against, from its own source.

    ``false`` is a boolean and needs no deriving. A string value the rollup does not compare
    against leaves the condition on default per-subject scoping while the ruleset reads as
    though it narrowed.
    """
    if "subject_scope_values" not in _CORPUS_CACHE:
        _CORPUS_CACHE["subject_scope_values"] = set(
            re.findall(r'subject_scope"\)[^\n]*?[=!]= "([a-z_]+)"', _verdict_source())
        )
    return _CORPUS_CACHE["subject_scope_values"]


def _check_subject_scope(
    cond: Dict[str, Any],
    cid: str,
    spec: Dict[str, Any],
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """A declared ``subject_scope`` narrowing is applied or it is an error.

    ``false`` (widen to all source rows) is a boolean and always applies. String values narrow
    and each has a precondition: ``element`` requires a ``subject_discovery`` with path, subject,
    and source, and the condition must read that source. Missing either, the condition falls back
    to the pooled reading the declaration exists to remove.
    """
    declared = cond.get("subject_scope")
    if declared is None or isinstance(declared, bool):
        return
    value = str(declared or "").strip().lower()
    values = subject_scope_values()
    if values and value not in values:
        diags.append(
            _diag(
                "error",
                "subject-scope-noop",
                f"{cid or '(unnamed)'}: `subject_scope: {declared}` is not a value the engine "
                "acts on, so this check keeps the default per-subject row scoping",
                path=rel,
                line=line,
                detail="accepted: false, " + ", ".join(sorted(values)),
                hint="`false` is the boolean, not the string",
            )
        )
        return
    if value != "element":
        return
    decl = spec.get("subject_discovery") or {}
    decl = decl if isinstance(decl, dict) else {}
    missing = [
        name
        for name, ok in (
            ("path", str(decl.get("path") or "").strip()),
            ("subject", [p for p in (decl.get("subject") or []) if str(p or "").strip()]),
            ("source", str(decl.get("source") or "").strip()),
        )
        if not ok
    ]
    if missing:
        diags.append(
            _diag(
                "error",
                "element-scope-undeclared",
                f"{cid or '(unnamed)'}: `subject_scope: element` has no element to narrow to — "
                f"this ruleset's subject_discovery declares no {', '.join(missing)}",
                path=rel,
                line=line,
                hint=(
                    "the element narrowing reuses the discovery walk, so the same `path` + "
                    "`subject` that finds the subjects is what tells one identity's entries "
                    "from another's"
                ),
            )
        )
        return
    reads: Set[str] = set()
    _walk_source_refs(cond, reads)
    if str(decl.get("source")) not in reads:
        diags.append(
            _diag(
                "error",
                "element-scope-unreachable",
                f"{cid or '(unnamed)'}: `subject_scope: element` cannot be applied — this check "
                f"reads {', '.join(sorted(reads)) or '(no source)'} and the subject's elements "
                f"are on {decl.get('source')!r}",
                path=rel,
                line=line,
                hint=(
                    "a condition on another source has no per-identity element to narrow to; "
                    "pair its two sides within one record (`pair_by`) instead"
                ),
            )
        )


def _check_pair_by(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """``pair_by`` is honoured or it is an error.

    Three ways it can be declared and do nothing: on a kind that ignores it, with a value
    the evaluator does not recognise, or when the two sides name different sources or rows.
    All three leave the comparison pooled while the pack reads as though it paired within a
    record.
    """
    declared = str(cond.get("pair_by", "") or "").strip()
    if not declared:
        return
    kinds, values = pair_by_kinds(), pair_by_values()
    if kinds and kind not in kinds:
        diags.append(
            _diag(
                "error",
                "pair-by-noop",
                f"{cid or '(unnamed)'}: `pair_by` is inert on kind {kind!r} — the "
                "comparison stays pooled across the source",
                path=rel,
                line=line,
                detail="honoured only by: " + ", ".join(sorted(kinds)),
                hint="only a two-sided kind has records to pair its sides within",
            )
        )
        return
    if values and declared.lower() not in values:
        diags.append(
            _diag(
                "error",
                "pair-by-noop",
                f"{cid or '(unnamed)'}: `pair_by: {declared}` is not a value the evaluator "
                "acts on, so the comparison stays pooled across the source",
                path=rel,
                line=line,
                detail="accepted: " + ", ".join(sorted(values)),
            )
        )
        return
    a = cond.get("left") or cond.get("start") or {}
    b = cond.get("right") or cond.get("end") or {}
    a = a if isinstance(a, dict) else {}
    b = b if isinstance(b, dict) else {}
    reason = ""
    if str(a.get("source") or "").strip() != str(b.get("source") or "").strip():
        reason = "the two sides name different sources, so no one record carries both"
    elif str(a.get("records") or "").strip() != str(b.get("records") or "").strip():
        reason = "the two sides declare different `records:` paths"
    elif (a.get("where") or []) != (b.get("where") or []):
        reason = "the two sides select different records through `where:`"
    if reason:
        diags.append(
            _diag(
                "error",
                "pair-by-unpairable",
                f"{cid or '(unnamed)'}: `pair_by` cannot be honoured — {reason}",
                path=rel,
                line=line,
                hint=(
                    "sides that deliberately select different records (an interval whose "
                    "ends are two versions of one entity) cannot be paired; drop the "
                    "declaration rather than leaving it to be refused at evaluation time"
                ),
            )
        )


def _src_python_text() -> str:
    """Every ``.py`` file under ``src/``, concatenated: the corpus for "does anything read this".

    Coarse heuristic: a pack key can reach the engine as a quoted ``.get("k")``, a bare Pydantic
    field, or through ``**`` expansion. Over-reporting a key that is read costs a reworded
    warning; missing one costs an inert declaration nobody questions.
    """
    if "src_py" not in _CORPUS_CACHE:
        _CORPUS_CACHE["src_py"] = "".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in sorted((REPO_ROOT / "src").rglob("*.py"))
        )
    return _CORPUS_CACHE["src_py"]


def _is_read_by_engine(key: str) -> bool:
    src = _src_python_text()
    if f'"{key}"' in src or f"'{key}'" in src:
        return True
    return bool(
        re.search(rf"(?:^\s+{re.escape(key)}\s*:|\.{re.escape(key)}\b)", src, re.M)
    )


def _neutrality_corpus() -> List[Tuple[str, List[str]]]:
    if "neutrality" not in _CORPUS_CACHE:
        out: List[Tuple[str, List[str]]] = []
        for root in _NEUTRALITY_ROOTS:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_dir() or path.suffix not in _NEUTRALITY_SUFFIXES:
                    continue
                rel = str(path.relative_to(REPO_ROOT))
                out.append(
                    (
                        rel,
                        path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    )
                )
        _CORPUS_CACHE["neutrality"] = out
    return _CORPUS_CACHE["neutrality"]


def vocabulary_pattern(words) -> Optional[re.Pattern]:
    """One alternation over ``words``, longest-first, with a custom boundary instead of ``\\b``.

    Duplicated from the suite's scanner, character for character. Python treats ``_`` as a
    word character, so ``\\b`` does not match a listed word *inside a snake_case identifier*,
    which is exactly where a domain noun is hardest to see on review. This boundary rejects
    only adjacent letters and digits, so a listed stem catches its compounds while an
    inflection it did not list still does not match.
    """
    words = [w for w in words if w]
    if not words:
        return None
    return re.compile(
        r"(?<![A-Za-z0-9])("
        + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        + r")(?![A-Za-z0-9])",
        re.I,
    )


# ------------------------------------------------------------------ diagnostics


def _diag(
    severity: str,
    code: str,
    message: str,
    *,
    path: str = "",
    line: int = 0,
    detail: str = "",
    hint: str = "",
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "path": path,
        "line": int(line or 0),
        "message": message,
        "detail": detail,
        "hint": hint,
    }


def _find_key_line(text: str, key: str) -> int:
    """Best-effort 1-indexed line of ``key:`` in ``text``; 0 when not found.

    Best-effort on purpose and documented as such: the same key name legitimately appears
    at several depths, and the first hit is a good enough place to put the operator's
    cursor. A precise answer would need a round-trip parser, which this module deliberately
    does not use; see ``pack_store`` on why nothing here re-serialises a pack file.
    """
    if not key:
        return 0
    m = re.search(rf"^\s*{re.escape(key)}\s*:", text, re.M)
    return text[: m.start()].count("\n") + 1 if m else 0


def _find_entry_key_line(text: str, name: str, key: str) -> int:
    """First ``key:`` line at or after the entry that declares ``name: <name>``.

    Same best-effort contract as :func:`_find_key_line`, one entry narrower. A catalog
    declares the same key on every source, so the file's first hit puts the operator's cursor
    on somebody else's entry, and a per-source diagnostic that always cites line 12 reads as
    one finding repeated rather than several distinct ones. Falls back to the file-wide hit
    (then 0) when the entry is spelled some other way, e.g. a key arriving through a YAML
    merge key rather than written in the entry.
    """
    if not name:
        return _find_key_line(text, key)
    anchor = re.search(
        rf"^\s*-?\s*name\s*:\s*['\"]?{re.escape(name)}['\"]?\s*(?:#.*)?$", text, re.M
    )
    if anchor:
        m = re.search(rf"^\s*{re.escape(key)}\s*:", text[anchor.start() :], re.M)
        if m:
            return text[: anchor.start() + m.start()].count("\n") + 1
    return _find_key_line(text, key)


def _find_id_line(text: str, cid: str) -> int:
    """Line where a condition declares itself, i.e. ``id: <cid>``.

    Separate from :func:`_find_key_line` because a condition's name is a *value*, not a
    key: searching for ``<the id>:`` finds nothing, and the operator gets line 0 for the
    diagnostics most likely to need a cursor.
    """
    if not cid:
        return 0
    # The `- ` is optional and load-bearing: a condition is a list item, so its id is
    # written `- id: x` at the start of an entry and `  id: x` on a later line. Requiring
    # the indent alone silently returned 0 for every first-line id.
    m = re.search(rf"^\s*(?:-\s+)?id:\s*[\"']?{re.escape(cid)}[\"']?\s*$", text, re.M)
    return text[: m.start()].count("\n") + 1 if m else 0


def _yaml_error_line(exc: Exception) -> int:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line = getattr(mark, "line", None)
    return int(line) + 1 if isinstance(line, int) else 0


def _is_anchor_error(exc: Exception) -> bool:
    """Is this parse failure an unresolved anchor/alias rather than bad syntax?

    Its own code because the blast radius differs. A syntax error is usually where the
    author just typed; a ``*ref`` whose ``&anchor`` was deleted fails the entire document,
    from a line that is often far from the edit, and the pack's biggest catalog is
    anchor-heavy, so this is the realistic way to empty it.
    """
    if isinstance(exc, yaml.composer.ComposerError):
        return True
    return "undefined alias" in str(exc).lower()


# --------------------------------------------------------------- file readers


def _load_yaml_strict(
    path: Path, rel: str, diags: List[Dict[str, Any]]
) -> Optional[Any]:
    """Parse one YAML file, reporting instead of swallowing. ``None`` on failure.

    The deliberate opposite of ``pack._read_yaml``, which is the thing being checked: it
    returns ``{}`` on any error so startup survives, and that ``{}`` is what makes a broken
    file look like an absent one.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        diags.append(
            _diag("error", "file-unreadable", f"{rel}: cannot be read: {exc}", path=rel)
        )
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        anchor = _is_anchor_error(exc)
        diags.append(
            _diag(
                "error",
                "yaml-anchor-unresolved" if anchor else "yaml-parse-failed",
                f"{rel}: does not parse, so the loader reads it as EMPTY",
                path=rel,
                line=_yaml_error_line(exc),
                detail=str(exc),
                hint=(
                    "an alias (*name) has no matching anchor (&name) — the whole file "
                    "fails, not just that line"
                    if anchor
                    else "fix the syntax at the line above"
                ),
            )
        )
        return None
    emptiness = _emptiness(text, data)
    if emptiness == "commented_out":
        # Warning, not error: a file holding only comments is ambiguous between a body
        # commented out during an edit and a new stub. As an error the editor would refuse to
        # save a new file until its first real key existed. The loader treats it as absent.
        diags.append(
            _diag(
                "warning",
                "yaml-empty-but-nonblank",
                f"{rel}: holds only comments, so the loader sees no facts in it",
                path=rel,
                hint="a stub, or a body commented out during an edit and never restored",
            )
        )
    elif emptiness == "content":
        # Error: real uncommented content that the loader still sees as nothing. The
        # typical cause is a top-level key the loader does not recognise.
        diags.append(
            _diag(
                "error",
                "yaml-empty-but-nonblank",
                f"{rel}: has content but parses to nothing, so the loader sees no facts",
                path=rel,
                hint=(
                    "the top-level key was probably renamed to one the loader does not read"
                ),
            )
        )
    return data


def _emptiness(text: str, data: Any) -> str:
    """``""`` when the document carries facts, else why it does not: the severity turns on it."""
    if data not in (None, {}, []):
        return ""
    saw_comment = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in ("---", "..."):
            continue
        if stripped.startswith("#"):
            saw_comment = True
            continue
        return "content"
    return "commented_out" if saw_comment else ""


def _check_frontmatter(path: Path, rel: str, diags: List[Dict[str, Any]]) -> None:
    """A Markdown doc's leading ``---`` block must parse. Absent is fine; broken is not.

    Broken matters because the block is a functional input, not decoration: a playbook's
    ``correlation:`` block and a concept's ``concept_id`` both live there, and
    ``_parse_frontmatter`` degrades to ``{}``, so a malformed block silently unnames the
    document and every reference to it becomes an orphan.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        diags.append(
            _diag("error", "file-unreadable", f"{rel}: cannot be read: {exc}", path=rel)
        )
        return
    if not text.startswith("---"):
        return
    end = text.find("\n---", 3)
    if end == -1:
        return
    try:
        yaml.safe_load(text[3:end])
    except yaml.YAMLError as exc:
        diags.append(
            _diag(
                "error",
                "frontmatter-parse-failed",
                f"{rel}: the frontmatter block does not parse, so its id and any "
                "structured block are dropped",
                path=rel,
                line=_yaml_error_line(exc),
                detail=str(exc),
            )
        )


def _doc_ids(docs_dir: Path, key: str) -> Set[str]:
    """Ids provided by a directory of Markdown docs (frontmatter id, else the file stem)."""
    out: Set[str] = set()
    if not docs_dir.is_dir():
        return out
    for path in sorted(docs_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        m = re.search(rf"^{re.escape(key)}:\s*(\S+)", text, re.M)
        out.add(m.group(1).strip("\"'") if m else path.stem)
    return out


# ---------------------------------------------------------------- sub-checks


def _walk_source_refs(node: Any, found: Set[str]) -> None:
    """Collect every logical-source name a ruleset reads, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _SOURCE_REF_KEYS:
                if isinstance(value, str):
                    found.add(value)
                elif isinstance(value, list):
                    found.update(v for v in value if isinstance(v, str))
                continue
            _walk_source_refs(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_source_refs(item, found)


# ------------------------------------------- field paths against the pack's own inventory

#: A candidate path is considered only when it is dotted. Single-segment values are enum
#: members or prose labels, not distinguishable from row paths without an engine map.
#: Dead single-segment paths are not caught except via ``_flattened_struct_prefixes``.
_DOTTED_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")

#: A single-segment value, the shape ``_DOTTED_PATH`` deliberately skips. Collected only to be
#: offered to the one rule that can decide it mechanically; never to the existence check.
_FLAT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Keys whose value is dotted but not a row path. Source-reference keys name a logical
#: source, ``use`` names a library entry, and ``records``/``array`` are read as an enclosing
#: prefix (their own value is checked through the children below them, not twice).
_NON_PATH_KEYS = frozenset(_SOURCE_REF_KEYS) | frozenset(
    {"use", "id", "kind", "use_case", "data", "records", "array", "from_entity"}
)

#: Keys whose value names a container intentionally: exempt from the container-naming check
#: but not the existence check (a misspelled container is as dead as a misspelled leaf).
#: Kept separate from ``_NON_PATH_KEYS`` so the existence check still applies.
_CONTAINER_KEYS = frozenset({"arrays"})

#: Blocks not descended at all. ``declares`` is the one that would otherwise misfire: its
#: ``field:`` values are entity-type names, not paths on a row.
_NON_PATH_BLOCKS = frozenset({"declares", "labels", "phrases", "sources"})

#: Both shapes `scripts/generate_source_schemas.py` writes.
_SCHEMA_TABLE_KEYS = ("fields", "columns")


def _schema_index(root: Path) -> Dict[str, Set[str]]:
    """``{source: paths}`` from the pack's own ``schemas/*.yaml``, cached on file mtimes.

    Read from the schema docs rather than from a retriever's discovered schema: discovery
    caps depth and leaf count, so a reported-absent path may be a real one the discovery
    missed. Cached on mtimes so a just-regenerated schema is never answered from a stale entry.
    Both generated shapes are read: the elasticsearch fields dict and the SQL columns/leaves.
    """
    schemas = root / "schemas"
    if not schemas.is_dir():
        return {}
    files = sorted(list(schemas.glob("*.yaml")) + list(schemas.glob("*.yml")))
    try:
        signature = tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files
        )
    except OSError:  # a file vanished mid-scan; fall through to an uncached read
        signature = None
    cache_key = ("schemas", str(root), signature)
    cached = _CORPUS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index: Dict[str, Set[str]] = {}
    for path in files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except yaml.YAMLError:
            continue  # reported by the parse pass; not this check's business
        if not isinstance(data, dict):
            continue
        source = str(data.get("source") or path.stem)
        paths: Set[str] = set()
        for table in (data.get("tables") or {}).values():
            if not isinstance(table, dict):
                continue
            fields = table.get("fields")
            if isinstance(fields, dict):
                paths |= {
                    str(name)
                    for name, spec in fields.items()
                    if not _is_null_only(spec)
                }
            columns = table.get("columns")
            if isinstance(columns, dict):
                for name, column in columns.items():
                    paths.add(str(name))
                    for leaf in (column or {}).get("leaves") or []:
                        if isinstance(leaf, dict) and leaf.get("path"):
                            paths.add(str(leaf["path"]))
        if paths:
            index.setdefault(source, set()).update(paths)
    if signature is not None:
        _CORPUS_CACHE[cache_key] = index
    return index


def _is_null_only(spec: Any) -> bool:
    """Is this generated leaf a key that only ever held ``null``?

    A null-only leaf is not a field a condition can read: resolution returns ``unknown``
    on every row for it, exactly as for an absent path. Qualifies only when every recorded
    type is ``null``; an optional field (``['null', 'string']``) stays, as does a leaf with no
    recorded types.
    """
    if not isinstance(spec, dict):
        return False
    types = spec.get("json_types")
    if not isinstance(types, list) or not types:
        return False
    return all(str(t).strip().lower() == "null" for t in types)


def _path_in_schema(paths: Set[str], candidate: str) -> bool:
    """Whether ``candidate`` would land somewhere in the documented paths.

    Four rules, each mirroring a step in the engine's own resolver:
    1. exact path;
    2. underscore-flattened alias at every prefix cut, remainder preserved;
    3. suffix under a named struct: a relative path under an enclosing ``records``/``array``;
    4. opaque prefix: a prefix recorded with no children is accepted as unknowable inside.
    """
    if candidate in paths:
        return True
    segments = candidate.split(".")
    for cut in range(len(segments), 0, -1):
        alias = "_".join(segments[:cut])
        rest = segments[cut:]
        if (".".join([alias, *rest]) if rest else alias) in paths:
            return True
    suffix = "." + candidate
    for known in paths:
        if known.endswith(suffix):
            return True
    for cut in range(len(segments) - 1, 0, -1):
        prefix = ".".join(segments[:cut])
        if prefix in paths and not any(
            known.startswith(prefix + ".") for known in paths
        ):
            return True
    return False


def _flattened_struct_prefixes(paths: Set[str]) -> Dict[str, str]:
    """``{flattened proper prefix: a recorded path it is a prefix of}``.

    The one sub-class of single-segment name whose deadness is decidable: a flat value that
    is the underscore-flattening of a recorded path's proper prefix provably names a container.
    The engine's rule 2 resolves it to the struct, which the evaluator then discards, so the
    check reads ``unknown`` on every row.

    Proper prefixes only: a full flattening of a recorded path is a leaf under rule 2 and is
    correct authoring against an aliased projection.
    """
    out: Dict[str, str] = {}
    for path in sorted(paths):
        segments = path.split(".")
        for cut in range(1, len(segments)):
            out.setdefault("_".join(segments[:cut]), path)
    return out


def _path_lists(
    node: Any,
    *,
    source: str,
    prefix: str,
    confirm_in: List[str],
    out: List[Tuple[str, List[str], str, List[str], List[str]]],
) -> None:
    """Collect ``(key, source names, prefix, dotted paths, flat names)`` per candidate list.

    A list and not a path because the engine ORs across a candidate list; one stale spelling
    beside live ones changes no outcome, so the defect is a whole list resolving to nothing.
    Dotted and flat names are kept separate: dotted names go to the existence check; flat
    names go only to ``_flattened_struct_prefixes``. ``confirm_fields`` are redirected to
    the sibling ``confirm_in`` sources where the evaluator resolves them.
    """
    if isinstance(node, dict):
        source = str(node.get("source") or source or "")
        prefix = str(node.get("records") or node.get("array") or "") or prefix
        confirm = [str(s) for s in (node.get("confirm_in") or [])] or confirm_in
        for key, value in node.items():
            name = str(key)
            if name in _NON_PATH_BLOCKS:
                continue
            targets = confirm if name == "confirm_fields" else [source]
            if isinstance(value, str):
                if name not in _NON_PATH_KEYS:
                    if _DOTTED_PATH.match(value):
                        out.append((name, targets, prefix, [value], []))
                    elif _FLAT_NAME.match(value):
                        out.append((name, targets, prefix, [], [value]))
                continue
            if isinstance(value, list):
                if name in _NON_PATH_KEYS:
                    continue
                paths: List[str] = []
                nested = False
                for item in value:
                    if isinstance(item, str):
                        paths.append(item)
                    elif isinstance(item, list):
                        paths.extend(v for v in item if isinstance(v, str))
                    else:
                        nested = True
                dotted = [p for p in paths if _DOTTED_PATH.match(p)]
                flat = [p for p in paths if _FLAT_NAME.match(p)]
                if dotted or flat:
                    out.append((name, targets, prefix, dotted, flat))
                if nested:
                    _path_lists(
                        value, source=source, prefix=prefix, confirm_in=confirm, out=out
                    )
                continue
            _path_lists(
                value, source=source, prefix=prefix, confirm_in=confirm, out=out
            )
    elif isinstance(node, list):
        for item in node:
            _path_lists(
                item, source=source, prefix=prefix, confirm_in=confirm_in, out=out
            )


def _resolve_renames(candidate: str, renames: Dict[str, str]) -> Optional[str]:
    """Return ``candidate`` with a leading projection alias replaced by the path it renames.

    ``None`` where the alias is pinned to no single path; a caller must read that as
    unknowable rather than absent.
    """
    if not renames:
        return candidate
    segments = candidate.split(".")
    for cut in range(len(segments), 0, -1):
        alias = ".".join(segments[:cut])
        if alias not in renames:
            continue
        target = renames[alias]
        if not target:
            return None
        return ".".join([target, *segments[cut:]])
    return candidate


def _resolves(
    path: str,
    *,
    prefix: str,
    known: Dict[str, Set[str]],
    renames: Dict[str, Dict[str, str]],
    exempt: Set[str],
) -> bool:
    """Whether ``path`` lands on any of the documented targets.

    Silence wins every tie: a name the projection cannot pin to one path is treated as
    resolved rather than absent, because unknowable is not the same as missing.
    """
    full = f"{prefix}.{path}" if prefix else path
    if any(c.startswith(e + ".") or c == e for e in exempt for c in (path, full)):
        return True
    for target, schema in known.items():
        rename = renames.get(target) or {}
        candidates = [_resolve_renames(c, rename) for c in (full, path)]
        if any(c is None for c in candidates):
            return True  # renamed by an entry naming no single path
        if any(_path_in_schema(schema, c) for c in candidates if c):
            return True
    return False


def _check_flattened_struct(
    flat: List[str],
    *,
    lands: Dict[str, Any],
    known: Dict[str, Set[str]],
    name: str,
    anchor: str,
    key: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Warn when every member of a candidate list provably names a container, not a leaf.

    A member with no struct witness is skipped; any member that resolves to a leaf is the
    whole list's answer (the engine ORs). The witness test outranks bare membership: a
    container recorded in the inventory passes ``_path_in_schema`` but resolves to the struct
    the evaluator discards, so only a witness for something under the container is conclusive.
    """
    if name in _CONTAINER_KEYS:
        return  # this key's value is expected to name a container
    witnesses: List[Tuple[str, str, str]] = []
    lands_on_a_leaf = False
    for path in flat:
        hit: Optional[Tuple[str, str, str]] = None
        for target, schema in sorted(known.items()):
            found = _flattened_struct_prefixes(schema).get(path)
            if found:
                hit = (path, target, found)
                break
        if hit is not None:
            witnesses.append(hit)
        elif _resolves(path, **lands):
            lands_on_a_leaf = True
    if lands_on_a_leaf or not witnesses:
        return
    named = ", ".join(f"`{p}` -> the struct inside `{hit}`" for p, _, hit in witnesses)
    diags.append(
        _diag(
            "warning",
            "field-name-flattens-a-struct",
            f"{anchor}: `{name}` names a CONTAINER and not a leaf — {named} on "
            f"{'/'.join(sorted(known))}; resolution lands on the struct, terminal-leaf "
            "extraction keeps none of it, and the check reads `unknown` on every row",
            path=rel,
            line=line,
            detail=(
                f"ruleset {key!r}; "
                + "; ".join(f"{p} flattens a prefix of {hit} on {t}" for p, t, hit in witnesses)
            ),
            hint=(
                "write the dotted path down to the LEAF this check means (the generated "
                "schemas/<source>.yaml lists them); a name that flattens a container reads "
                "as a valid field and binds nothing, which is what the source being rebound "
                "from a flat shape onto a nested one leaves behind"
            ),
        )
    )


def _check_field_paths(
    blocks: List[Tuple[str, Any, int]],
    *,
    key: str,
    rel: str,
    root: Path,
    declared: Dict[str, str],
    physical: Set[str],
    decoded: Dict[str, List[str]],
    renames: Dict[str, Dict[str, str]],
    diags: List[Dict[str, Any]],
) -> int:
    """Warn when no path in a candidate list exists on the documented target schema.

    A warning, not an error: an ES inventory is sampled, a SQL query may project an alias
    (e.g. ``SELECT `a`.`b` AS c`` in ``query_hints``), and a pack documents only the sources
    it chose to, so absence of an inventory is not evidence about a path. Silent wherever there is no schema doc for the target. Declared projection
    aliases are resolved back through ``renames`` before comparison. A decidable single-segment
    struct case is also checked (see ``_check_flattened_struct``), but does not add to the
    returned count. Returns how many candidate lists could be checked.
    """
    index = _schema_index(root)
    if not index:
        return 0
    checked = 0
    for anchor, node, line in blocks:
        found: List[Tuple[str, List[str], str, List[str], List[str]]] = []
        _path_lists(node, source="", prefix="", confirm_in=[], out=found)
        for name, logicals, prefix, paths, flat in found:
            targets = {
                str(declared.get(logical) or (logical if logical in physical else ""))
                for logical in logicals
                if logical
            }
            known = {t: index[t] for t in targets if t in index}
            if not known:
                continue  # no documented source behind this list; silence, not a pass
            exempt = {p for t in known for p in (decoded.get(t) or [])}
            lands = {
                "prefix": prefix,
                "known": known,
                "renames": renames,
                "exempt": exempt,
            }
            if not paths:
                _check_flattened_struct(
                    flat,
                    lands=lands,
                    known=known,
                    name=name,
                    anchor=anchor,
                    key=key,
                    rel=rel,
                    line=line,
                    diags=diags,
                )
                continue
            checked += 1
            if any(_resolves(path, **lands) for path in paths):
                continue
            diags.append(
                _diag(
                    "warning",
                    "field-path-not-in-schema",
                    f"{anchor}: not one of the {len(paths)} path(s) in `{name}` exists on "
                    f"{'/'.join(sorted(known))} — the check can only read `unknown`, which "
                    "in the report is indistinguishable from a source that had no rows",
                    path=rel,
                    line=line,
                    detail=(
                        f"ruleset {key!r}; paths: {', '.join(paths)}"
                        + (f"; relative to {prefix!r}" if prefix else "")
                    ),
                    hint=(
                        "check the spelling against the pack's own schemas/<source>.yaml; "
                        "if it is a query_hints projection alias or a field the sample "
                        "missed, the path is fine and this is noise"
                    ),
                )
            )
    return checked


#: Endpoint kinds where a source's ``projection`` is the required ``SELECT`` list.
#: On every other kind the key is inert. ``test_pack_validate.py`` pins both ends
#: so a new backend gaining this feature must update this set.
_PROJECTION_KINDS = frozenset({"databricks_uc"})


def _projected_paths(catalog: Any) -> Dict[str, List[str]]:
    """``{source: projected row names}`` for sources whose kind enforces a projection.

    Names, not projection entries: an entry may be an expression whose row column is the
    declared ``AS`` alias, not the expression text. An expression with no alias contributes
    nothing; contributing the raw text would report every condition reading such a leaf as
    uncovered.
    """
    out: Dict[str, List[str]] = {}
    for entry in (catalog or {}).get("sources") or []:
        if not isinstance(entry, dict):
            continue
        endpoints = entry.get("endpoints")
        kind = str(
            (endpoints.get("kind") if isinstance(endpoints, dict) else "") or ""
        ).strip()
        if kind.lower() not in _PROJECTION_KINDS:
            continue
        paths = [
            name
            for c in (entry.get("projection") or [])
            if c
            for name in projection_names(str(c))
        ]
        if paths:
            out[str(entry.get("name", "") or "")] = paths
    return out


def _projection_renames(catalog: Any) -> Dict[str, Dict[str, str]]:
    """``{source: {alias: the path it renames}}`` for use by ``_check_field_paths``.

    Not filtered to projection kinds: a declared ``AS`` renames on any backend, and missing
    a rename on a non-enforcing backend costs nothing while reporting one turns a correct
    path into a finding.
    """
    out: Dict[str, Dict[str, str]] = {}
    for entry in (catalog or {}).get("sources") or []:
        if not isinstance(entry, dict):
            continue
        mapping = projection_renames(
            [str(c) for c in (entry.get("projection") or []) if c]
        )
        if mapping:
            out[str(entry.get("name", "") or "")] = mapping
    return out


def _check_projection_names(
    entry: Dict[str, Any],
    *,
    rel: str,
    text: str,
    diags: List[Dict[str, Any]],
) -> int:
    """Warn or error when two projection entries would arrive in a row under one name.

    Worse than a missing field: the row is populated, but one entry's value is silently served
    under the other's name. Error when every colliding entry pins its own ``AS`` (the row
    provably cannot carry both); warning when at least one falls back to its last segment (the
    convention may alias them apart). Restricted to kinds that read ``projection`` as
    authoritative. Returns 0 or 1.
    """
    endpoints = entry.get("endpoints")
    kind = str(
        (endpoints.get("kind") if isinstance(endpoints, dict) else "") or ""
    ).strip()
    if kind.lower() not in _PROJECTION_KINDS:
        return 0
    entries = [str(c) for c in (entry.get("projection") or []) if c]
    if not entries:
        return 0
    name = str(entry.get("name", "") or "") or "?"
    for row_name, shared in sorted(colliding_row_names(entries).items()):
        distinct = sorted(set(shared))
        pinned = [e for e in distinct if projection_alias(e)]
        every = len(pinned) == len(distinct)
        diags.append(
            _diag(
                "error" if every else "warning",
                "projection-name-collision",
                f"source {name!r}: {len(distinct)} projection entries arrive under the one "
                f"row name {row_name!r}"
                + (
                    ", and each pins that name itself — the row cannot carry both, so one "
                    "entry's value is silently served under the other's name"
                    if every
                    else ", so unless every one of them is aliased apart the row keeps only "
                    "the last and a check reading either is answered with one value"
                ),
                path=rel,
                line=_find_entry_key_line(text, name, "projection"),
                detail=f"entries: {'; '.join(distinct)}",
                hint=(
                    "give each entry its own `AS <name>` — the flattened path is the "
                    "convention the generator is told to apply, so aliasing to it changes "
                    "nothing where the generator already obeys and fixes it where it does not"
                ),
            )
        )
    return 1


def _check_projection_coverage(
    blocks: List[Tuple[str, Any, int]],
    *,
    key: str,
    rel: str,
    root: Path,
    declared: Dict[str, str],
    physical: Set[str],
    projected: Dict[str, List[str]],
    renames: Dict[str, Dict[str, str]],
    decoded: Dict[str, List[str]],
    diags: List[Dict[str, Any]],
) -> int:
    """Warn when a schema-documented path will not appear in the row because it is not projected.

    ``_check_field_paths`` checks whether the leaf exists; this checks whether it will be
    retrieved. A source with a ``projection:`` hands its generator a required ``SELECT`` list,
    so a documented leaf outside that list never arrives. Silent wherever there is no schema
    doc for the target, and for backends that ignore ``projection``. Returns the count of path
    lists checked.
    """
    if not projected:
        return 0
    index = _schema_index(root)
    if not index:
        return 0
    checked = 0
    for anchor, node, line in blocks:
        found: List[Tuple[str, List[str], str, List[str], List[str]]] = []
        _path_lists(node, source="", prefix="", confirm_in=[], out=found)
        for name, logicals, prefix, paths, _flat in found:
            if not paths:
                continue  # a single-segment name is not a path; `_check_field_paths` owns it
            targets = {
                str(declared.get(logical) or (logical if logical in physical else ""))
                for logical in logicals
                if logical
            }
            for target in sorted(t for t in targets if t in projected and t in index):
                known = index[target]
                exempt = decoded.get(target) or []
                # Decoded field parts are synthesized from a projected column, so they are
                # exempt here as in the schema check; only the encoded column's coverage matters.
                aliases = renames.get(target) or {}
                roots = {
                    aliases.get(entry, entry) or entry for entry in projected[target]
                }
                inventory = (
                    {
                        p
                        for entry in roots
                        for p in known
                        if p == entry or p.startswith(entry + ".")
                    }
                    | roots
                    | set(projected[target])
                )
                checked += 1
                resolved = False
                for path in paths:
                    full = f"{prefix}.{path}" if prefix else path
                    if any(
                        c.startswith(e + ".") or c == e
                        for e in exempt
                        for c in (path, full)
                    ):
                        resolved = True
                        break
                    if _path_in_schema(inventory, full) or _path_in_schema(
                        inventory, path
                    ):
                        resolved = True
                        break
                if resolved:
                    continue
                diags.append(
                    _diag(
                        "warning",
                        "field-path-outside-projection",
                        f"{anchor}: none of the {len(paths)} path(s) in `{name}` is covered "
                        f"by {target}'s declared `projection`, so the leaf exists on the "
                        "schema and still never arrives in a row — the check reads "
                        "`unknown` on every run while every stage reports success",
                        path=rel,
                        line=line,
                        detail=(
                            f"ruleset {key!r}; paths: {', '.join(paths)}"
                            + (f"; relative to {prefix!r}" if prefix else "")
                        ),
                        hint=(
                            "add the leaf to that source's `projection:` — measuring the "
                            "scan cost first, as the projection's own note requires — or "
                            "read it from a narrower source; a leaf under an ALREADY "
                            "projected struct needs nothing"
                        ),
                    )
                )
    return checked


def _check_binding_paths(
    entry: Dict[str, Any],
    *,
    rel: str,
    text: str,
    index: Dict[str, Set[str]],
    decoded: Dict[str, List[str]],
    diags: List[Dict[str, Any]],
) -> int:
    """Warn when no path in an ``entity_bindings`` entry exists on the source's schema.

    A binding is a candidate list; the engine uses the first confirmed path. An entirely dead
    list means the retriever logs a stale-binding warning and runs unscoped for that entity
    type. Warning and silent for the same reasons as ``_check_field_paths``. Returns how many
    bindings were checked.
    """
    name = str(entry.get("name", "") or "") or "?"
    bindings = entry.get("entity_bindings")
    if not isinstance(bindings, dict) or not bindings:
        return 0
    schema = index.get(name)
    if not schema:
        return 0  # no documented source behind this binding; silence, not a pass
    exempt = [e for e in (decoded.get(name) or []) if e]
    checked = 0

    def _one(label: str, declared: Any) -> None:
        nonlocal checked
        paths = [str(p) for p in (declared or []) if str(p).strip()]
        if not isinstance(declared, list) or not paths:
            return
        checked += 1
        for path in paths:
            if any(path == e or path.startswith(e + ".") for e in exempt):
                return
            if _path_in_schema(schema, path):
                return
        diags.append(
            _diag(
                "warning",
                "binding-paths-not-in-schema",
                f"source {name!r}: not one of the {len(paths)} path(s) bound to "
                f"`{label}` exists on its schema — the retriever cannot filter on it, so "
                "the query runs unscoped for that entity type while the source still "
                "advertises it",
                path=rel,
                line=_find_entry_key_line(text, name, "entity_bindings"),
                detail=f"paths: {', '.join(paths)}",
                hint=(
                    "re-measure the source's own field names: a live spelling is a "
                    "one-line catalog fix, and no live spelling means the binding and "
                    "its `entities` entry should go rather than bind nothing"
                ),
            )
        )

    for etype, value in bindings.items():
        if isinstance(value, dict):
            for form, fields in value.items():
                _one(f"{etype}.{form}", fields)
        else:
            _one(str(etype), value)
    return checked


def _decoded_table_paths(catalog: Any) -> Dict[str, List[str]]:
    """``{source: [decode target path]}`` from the raw catalog, via ``encoded_fields`` itself.

    Decode targets are absent from every backend-discovered inventory by construction, so
    paths under them must be exempt from the schema check. The derivation delegates to the
    module that owns the declaration so the two cannot drift.
    """
    sources = [s for s in ((catalog or {}).get("sources") or []) if isinstance(s, dict)]
    if not any(s.get("encoded_fields") for s in sources):
        return {}
    shim = SimpleNamespace(
        sources=[
            SimpleNamespace(
                name=str(s.get("name", "") or ""),
                encoded_fields=s.get("encoded_fields") or [],
            )
            for s in sources
        ]
    )
    by_name = {s.name: s for s in shim.sources}
    shim.source = by_name.get  # type: ignore[attr-defined]
    # A malformed declaration is not this check's finding, and must not cost the whole lint.
    try:
        return decoded_tables(shim)
    except Exception:  # noqa: BLE001
        return {}


_STOPWORDS = frozenset(
    "a an the of on in to is are was were be been by for with and or this that it its "
    "at from as their per not no".split()
)
_NEGATIONS = re.compile(
    r"(?<![a-z])(not|no|never|without|neither|nor|non|absent|missing|lacks|lacking)"
    r"(?![a-z])|(?<![a-z])non-",
    re.I,
)

#: Minimum vocabulary overlap between a label and its requirement for the two to be treated
#: as the same sentence. Set where true positives and correct labels separate cleanly.
_LABEL_ECHO_COVERAGE = 0.7


def _content_words(text: Any) -> Set[str]:
    return {
        w
        for w in re.findall(r"[a-z]+", str(text or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def _label_echoes_requirement(label: Any, expected: Any) -> bool:
    """Whether the label restates the requirement rather than stating the finding.

    Two signals work together: shared vocabulary above ``_LABEL_ECHO_COVERAGE`` and the same
    count of negation words. Each signal alone over-fires in opposite directions; together they
    separate a label that shares the requirement's vocabulary and polarity from one that merely
    shares the subject. One-directional and silent where either signal has no content.
    """
    label_words = _content_words(label)
    expected_words = _content_words(expected)
    if not label_words or not expected_words:
        return False
    coverage = len(label_words & expected_words) / len(label_words)
    if coverage < _LABEL_ECHO_COVERAGE:
        return False
    return len(_NEGATIONS.findall(str(label or ""))) == len(
        _NEGATIONS.findall(str(expected or ""))
    )


# ------------------------------------------------------------------- the entry point


def validate_pack(pack_dir) -> Dict[str, Any]:
    """Every diagnostic for one pack directory.

    Returns ``{pack, ok, errors, warnings, infos, diagnostics, counts}``. ``ok`` is
    ``errors == 0``; warnings are reported and never block, so a caller gating a write on
    this reads ``ok`` and shows the rest.

    Takes a directory, not a pack name: the checked-in template lives outside the packs
    root and has to be validatable too, or the one pack every author starts from is the one
    pack nobody ever lints.
    """
    root = Path(pack_dir)
    diags: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    if not root.is_dir():
        return {
            "pack": root.name,
            "ok": False,
            "errors": 1,
            "warnings": 0,
            "infos": 0,
            "diagnostics": [
                _diag("error", "pack-missing", f"no pack directory at {root}")
            ],
            "counts": counts,
        }

    # --- 1. every file parses at all -----------------------------------------
    yaml_docs: Dict[str, Any] = {}
    n_files = 0
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts) or not path.is_file():
            continue
        n_files += 1
        rel = str(PurePosixPath(path.relative_to(root)))
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            yaml_docs[rel] = _load_yaml_strict(path, rel, diags)
        elif suffix == ".md":
            _check_frontmatter(path, rel, diags)
    counts["files"] = n_files
    counts["yaml_files"] = len(yaml_docs)

    _check_vocabulary(root, diags, counts)
    _check_glossary(root, yaml_docs, diags, counts)
    _check_catalog(root, yaml_docs, diags, counts)
    # Reads the results of both checks above, so it runs after both.
    _check_form_bindings(root, yaml_docs, diags, counts)
    _check_data_stems(root, diags, counts)
    # Validated before the rulesets so a condition naming a form can be checked against the
    # forms that actually loaded, rather than against the file's text.
    _equivalence_forms = _check_equivalence_forms(root, yaml_docs, diags, counts)
    _check_rulesets(root, yaml_docs, diags, counts, _equivalence_forms)
    _check_correlation_specs(root, diags, counts)
    _check_use_case_shape(root, diags, counts)

    errors = sum(1 for d in diags if d["severity"] == "error")
    warnings = sum(1 for d in diags if d["severity"] == "warning")
    infos = sum(1 for d in diags if d["severity"] == "info")
    order = {"error": 0, "warning": 1, "info": 2}
    diags.sort(
        key=lambda d: (order.get(d["severity"], 3), d["path"], d["line"], d["code"])
    )
    return {
        "pack": root.name,
        "ok": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "diagnostics": diags,
        "counts": counts,
    }


def _check_vocabulary(
    root: Path, diags: List[Dict[str, Any]], counts: Dict[str, int]
) -> None:
    """The pack declares its domain vocabulary and no word of it appears in the engine.

    Both halves fail a suite-wide test. The second fails retroactively: installing a pack
    that claims a word the engine already uses turns a green tree red without anything in
    the engine changing. A word the engine genuinely owns should be omitted from the
    vocabulary file with the reason recorded there.
    """
    path = root / VOCABULARY_FILE
    words: List[str] = []
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            words = [
                str(w).strip().lower()
                for w in (data.get("domain_vocabulary") or [])
                if str(w).strip()
            ]
        except yaml.YAMLError:
            pass  # already reported by the parse pass
    counts["vocabulary"] = len(words)
    if not words:
        diags.append(
            _diag(
                "error",
                "missing-domain-vocabulary",
                f"{VOCABULARY_FILE} declares no words, so nothing keeps this domain's "
                "nouns out of the engine",
                path=VOCABULARY_FILE,
                hint="list the domain's own nouns; a stem also covers its compounds",
            )
        )
        return
    pattern = vocabulary_pattern([w for w in words if w not in _ENGINE_OWNS])
    if pattern is None:
        return
    vocab_text = path.read_text(encoding="utf-8", errors="replace")
    seen: Set[str] = set()
    for rel, lines in _neutrality_corpus():
        for lineno, line in enumerate(lines, 1):
            for match in pattern.finditer(line):
                word = match.group(0).lower()
                if word in _ENGINE_OWNS or word in seen:
                    continue
                seen.add(word)
                diags.append(
                    _diag(
                        "error",
                        "neutrality-collision",
                        # Named as the pack declared it: quoting the engine's casing
                        # would send the operator searching for a word not in their list.
                        f"this pack claims {word!r}, which the engine already uses — "
                        "installing it makes the engine read as owning this domain",
                        path=VOCABULARY_FILE,
                        line=_find_word_line(vocab_text, word),
                        detail=f"{rel}:{lineno}: {line.strip()[:120]}",
                        hint=(
                            "either reword the engine so it does not name this, or drop "
                            "the word from this list and record why the engine owns it"
                        ),
                    )
                )


def _find_word_line(text: str, word: str) -> int:
    for lineno, line in enumerate(text.splitlines(), 1):
        if word in line.lower():
            return lineno
    return 0


def _check_glossary(
    root: Path,
    yaml_docs: Dict[str, Any],
    diags: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> None:
    rel = "entity_glossary.yaml"
    doc = yaml_docs.get(rel)
    entities = (doc or {}).get("entities") or [] if isinstance(doc, dict) else []
    counts["entities"] = len(entities)
    present = (root / rel).is_file()
    if not entities:
        # Absent and empty produce the same finding under one code. Reporting absence
        # explicitly matters because the editor can delete a file.
        diags.append(
            _diag(
                "warning",
                "no-entities-declared",
                (
                    f"{rel}: no entities, so nothing is extracted from an incident by type"
                    if present
                    else f"{rel} does not exist, so nothing is extracted by type"
                ),
                path=rel,
            )
        )
    if not present:
        return
    _report_unread_keys(
        doc if isinstance(doc, dict) else {}, rel, diags, where="the glossary"
    )
    text = (root / rel).read_text(encoding="utf-8", errors="replace")
    known = set(EntityDef.model_fields)
    for entry in entities:
        if not isinstance(entry, dict):
            continue
        for key in sorted(set(entry) - known):
            diags.append(
                _diag(
                    "warning",
                    "unknown-model-key",
                    f"entity {entry.get('type', '?')!r} declares {key!r}, which "
                    "EntityDef drops silently",
                    path=rel,
                    line=_find_key_line(text, key),
                    detail=f"known keys: {', '.join(sorted(known))}",
                    hint="a typo, or a key that still has to be wired into EntityDef",
                )
            )
        _check_value_form_stems(entry, rel, text, diags)


def _check_value_form_stems(
    entry: Dict[str, Any], rel: str, text: str, diags: List[Dict[str, Any]]
) -> None:
    """An unusable ``value_forms[].stem`` is an error: the widening guard is never handed a stem.

    An unusable stem is either an invalid regex or one with no capture group (exactly one is
    required). Either way ``value_stem`` returns ``None`` for every value and the widening is
    never applied.
    """
    for form in entry.get("value_forms") or []:
        if not isinstance(form, dict):
            continue
        stem = str(form.get("stem") or "").strip()
        if not stem:
            continue
        where = f"entity {entry.get('type', '?')!r} form {form.get('name', '?')!r}"
        try:
            compiled = re.compile(stem)
        except re.error as exc:
            diags.append(
                _diag(
                    "error",
                    "unusable-value-form-stem",
                    f"{where} declares a stem that is not a valid regex: {exc}",
                    path=rel,
                    line=_find_key_line(text, "stem"),
                    hint="the stem is matched against the value; an invalid pattern is "
                    "logged once and then ignored for every value",
                )
            )
            continue
        if compiled.groups != 1:
            diags.append(
                _diag(
                    "error",
                    "unusable-value-form-stem",
                    f"{where} declares a stem with {compiled.groups} capture groups; "
                    "exactly one is required, and it is the part a source may store "
                    "INSTEAD of the whole value",
                    path=rel,
                    line=_find_key_line(text, "stem"),
                    hint="wrap the identity core in parentheses, e.g. `^(<core>)<optional>`",
                )
            )


def _check_form_bindings(
    root: Path,
    yaml_docs: Dict[str, Any],
    diags: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> None:
    """Check that sources agree about whether entity form decides column routing.

    A form name the glossary does not declare is an error: no value can classify as it, so
    every value of the form it was meant to catch is dropped. Warning when some sources use
    per-form bindings and others stay flat: on the flat ones every form's value is filtered
    onto every listed column, which is a predicate matching nothing for the wrong form. Silent
    when all sources agree (all per-form or all flat).
    """
    cat_rel, glo_rel = "source_catalog.yaml", "entity_glossary.yaml"
    cat = yaml_docs.get(cat_rel)
    glo = yaml_docs.get(glo_rel)
    sources = (cat or {}).get("sources") or [] if isinstance(cat, dict) else []
    entities = (glo or {}).get("entities") or [] if isinstance(glo, dict) else []
    declared: Dict[str, List[str]] = {}
    for entry in entities:
        if not isinstance(entry, dict):
            continue
        names = [
            str(form.get("name") or "").strip()
            for form in entry.get("value_forms") or []
            if isinstance(form, dict)
        ]
        names = [n for n in names if n]
        if names:
            declared[str(entry.get("type") or "")] = names
    # Both counts always: an unchecked pack must not read like a checked one. The pair
    # distinguishes two states: no entity declares a form, versus none of the sources bind one.
    counts["value_form_entities"] = len(declared)
    counts["value_form_bindings_checked"] = 0
    if not declared or not sources:
        return
    cat_text = (
        (root / cat_rel).read_text(encoding="utf-8", errors="replace")
        if (root / cat_rel).is_file()
        else ""
    )
    per_form: Dict[str, List[str]] = {}
    flat: Dict[str, List[str]] = {}
    checked = 0
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "") or "") or "?"
        bindings = entry.get("entity_bindings")
        if not isinstance(bindings, dict):
            continue
        for etype, value in bindings.items():
            etype = str(etype)
            if etype not in declared:
                continue
            checked += 1
            if not isinstance(value, dict):
                flat.setdefault(etype, []).append(name)
                continue
            per_form.setdefault(etype, []).append(name)
            unknown = [str(k) for k in value if str(k) not in declared[etype]]
            if not unknown:
                continue
            diags.append(
                _diag(
                    "error",
                    "unknown-value-form-binding",
                    f"source {name!r} binds {etype} under form(s) "
                    f"{', '.join(sorted(unknown))}, which the glossary does not declare — "
                    "no value can classify as one, so those fields are never filtered on "
                    "and any value of the form they were meant to catch is dropped",
                    path=cat_rel,
                    line=_find_entry_key_line(cat_text, name, "entity_bindings"),
                    detail=f"declared forms: {', '.join(declared[etype])}",
                    hint=(
                        "spell the form as the glossary does, or add the form to "
                        f"`{glo_rel}` if it is real"
                    ),
                )
            )
    counts["value_form_bindings_checked"] = checked
    for etype, flat_sources in sorted(flat.items()):
        if etype not in per_form:
            continue
        for name in flat_sources:
            diags.append(
                _diag(
                    "warning",
                    "value-form-binding-mixed",
                    f"source {name!r} binds {etype} as a FLAT list while "
                    f"{len(per_form[etype])} other source(s) bind it per form — so here "
                    "every form's value is filtered onto every listed column, and a value "
                    "of the wrong form for that column is a predicate matching nothing",
                    path=cat_rel,
                    line=_find_entry_key_line(cat_text, name, "entity_bindings"),
                    detail=(
                        f"per-form: {', '.join(sorted(per_form[etype]))}; "
                        f"declared forms: {', '.join(declared[etype])}"
                    ),
                    hint=(
                        "measure which form(s) each column really stores, then bind "
                        "`{<form>: [<its columns>]}` — a column holding both forms is bound "
                        "under both, and a flat list left deliberately belongs in the "
                        "source's own notes so the next reader knows it was measured"
                    ),
                )
            )


def _check_catalog(
    root: Path,
    yaml_docs: Dict[str, Any],
    diags: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> None:
    rel = "source_catalog.yaml"
    doc = yaml_docs.get(rel)
    sources = (doc or {}).get("sources") or [] if isinstance(doc, dict) else []
    counts["sources"] = len(sources)
    present = (root / rel).is_file()
    if not sources:
        # Same as the glossary: a missing catalog and an empty one both load as a pack with
        # zero sources. Returning early on absence would have silenced the one edit that
        # produces this state in a single click.
        diags.append(
            _diag(
                "warning",
                "no-sources-declared",
                (
                    f"{rel}: no sources, so every condition resolves to `unknown`"
                    if present
                    else f"{rel} does not exist, so every condition resolves to `unknown`"
                ),
                path=rel,
                hint="this is the state a broken catalog also produces — see the errors",
            )
        )
    if not present:
        return
    _report_unread_keys(
        doc if isinstance(doc, dict) else {}, rel, diags, where="the catalog"
    )
    text = (root / rel).read_text(encoding="utf-8", errors="replace")
    known = set(SourceDef.model_fields)
    names: Dict[str, int] = {}
    redirects: List[tuple] = []
    index = _schema_index(root)
    decoded = _decoded_table_paths(doc if isinstance(doc, dict) else {})
    n_bindings = 0
    n_projections = 0
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "") or "")
        names[name] = names.get(name, 0) + 1
        for item in entry.get("not_answered_by") or []:
            if isinstance(item, dict):
                redirects.append((name, str(item.get("ask_instead", "") or "").strip()))
        _check_actor_key(entry, rel=rel, text=text, diags=diags)
        n_bindings += _check_binding_paths(
            entry,
            rel=rel,
            text=text,
            index=index,
            decoded=decoded,
            diags=diags,
        )
        n_projections += _check_projection_names(entry, rel=rel, text=text, diags=diags)
        for key in sorted(set(entry) - known):
            diags.append(
                _diag(
                    "warning",
                    "unknown-model-key",
                    f"source {name or '?'!r} declares {key!r}, which SourceDef drops "
                    "silently",
                    path=rel,
                    line=_find_key_line(text, key),
                    detail=f"known keys: {', '.join(sorted(known))}",
                    hint="a typo, or a key that still has to be wired into SourceDef",
                )
            )
    for name, n in sorted(names.items()):
        if n > 1:
            diags.append(
                _diag(
                    "error",
                    "duplicate-source-name",
                    f"{name!r} is declared {n} times; a lookup by name reaches only one "
                    "of them",
                    path=rel,
                    line=_find_key_line(text, "name"),
                )
            )
    counts["entity_bindings_checked"] = n_bindings
    counts["projections_checked"] = n_projections
    _check_redirect_targets(rel, text, redirects, set(names), diags)


# ``ask_instead`` is emitted verbatim, so a redirect naming an absent source sends the
# planner nowhere silently. This check fires only on a bare token (one word, no whitespace)
# because only that shape is an unambiguous claim that the named source exists.
_REDIRECT_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]{2,}$")


def _check_redirect_targets(
    rel: str,
    text: str,
    redirects: List[tuple],
    names: set,
    diags: List[Dict[str, Any]],
) -> None:
    """Warn on a `not_answered_by.ask_instead` naming a source the catalog does not declare."""
    for source, target in redirects:
        stripped = target.rstrip(".,;:")
        if not stripped or not _REDIRECT_TOKEN.match(stripped):
            # Empty, or prose. Prose is a legitimate authoring choice here and cannot be
            # checked mechanically, which is the whole reason this is a warning and not the
            # membership test it looks like it should be.
            continue
        if stripped in names:
            continue
        diags.append(
            _diag(
                "warning",
                "redirect-to-unknown-source",
                f"source {source!r} redirects a question it cannot answer to "
                f"{stripped!r}, which this catalog does not declare — the planner is "
                "told to ask something that does not exist",
                path=rel,
                line=_find_key_line(text, "ask_instead"),
                hint=(
                    "use the catalog name, or say in prose that no source here answers "
                    "it (a sentence is not checked and is a valid answer)"
                ),
            )
        )


# An actor key resolving to nothing causes queries to run unscoped for that entity type.
# Error when the key shape is unreadable; warning when a member is unbound or a type
# appears twice (resolves as the weaker one-column claim).
def _check_actor_key(
    entry: Dict[str, Any],
    *,
    rel: str,
    text: str,
    diags: List[Dict[str, Any]],
) -> None:
    """`identity_keys` / `require_all_entities` on one source: can they resolve at all?"""
    name = str(entry.get("name", "") or "") or "?"
    bound = set(entry.get("entity_bindings") or {})
    raw_keys = entry.get("identity_keys")
    candidates = raw_keys if isinstance(raw_keys, list) else []
    if raw_keys is not None and not isinstance(raw_keys, list):
        diags.append(
            _diag(
                "error",
                "identity-keys-not-a-list",
                f"source {name!r} declares `identity_keys` as "
                f"{type(raw_keys).__name__}, not a list of candidate keys, so the resolver "
                "reads no candidate at all and the query is not constrained to one actor",
                path=rel,
                line=_find_key_line(text, "identity_keys"),
                hint="a LIST of LISTS: `identity_keys: [[type_a, type_b]]`",
            )
        )
    line = _find_key_line(text, "identity_keys")
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, list):
            diags.append(
                _diag(
                    "error",
                    "identity-key-not-a-list",
                    f"source {name!r}: `identity_keys` candidate {index} is "
                    f"{type(candidate).__name__} ({candidate!r}), not a list of entity-type "
                    "names — it names no type the resolver can look up, so this candidate can "
                    "never be satisfied by any incident",
                    path=rel,
                    line=line,
                    hint="wrap it: `- [type_a, type_b]`, or `- [type_a]` for a one-column key",
                )
            )
            continue
        named = [m for m in candidate if isinstance(m, str) and m.strip()]
        for member in candidate:
            if isinstance(member, str) and member.strip():
                continue
            diags.append(
                _diag(
                    "error",
                    "identity-key-member-not-a-name",
                    f"source {name!r}: `identity_keys` candidate {index} carries "
                    f"{member!r} ({type(member).__name__}), which cannot name an entity "
                    "type — the member is dropped, so the candidate resolves a key the pack "
                    "did not declare, or none at all",
                    path=rel,
                    line=line,
                    hint="every member is an entity-type name, as a plain string",
                )
            )
        if not named:
            continue
        if len(set(named)) < len(named):
            diags.append(
                _diag(
                    "warning",
                    "identity-key-repeats-a-type",
                    f"source {name!r}: `identity_keys` candidate {index} names "
                    f"{', '.join(named)} — the same entity type more than once, so it "
                    "resolves to ONE column and makes the weaker one-column claim instead of "
                    "the conjunction two names ask for",
                    path=rel,
                    line=line,
                    hint="two DIFFERENT entity types, or drop the repeat and mean one column",
                )
            )
        _check_key_members_bound(
            named,
            f"`identity_keys` candidate {index}",
            name=name,
            bound=bound,
            rel=rel,
            line=line,
            diags=diags,
        )
    fixed = entry.get("require_all_entities")
    if isinstance(fixed, list):
        named = [m for m in fixed if isinstance(m, str) and m.strip()]
        fixed_line = _find_key_line(text, "require_all_entities")
        if named and len(set(named)) < 2:
            diags.append(
                _diag(
                    "warning",
                    "require-all-entities-not-a-conjunction",
                    f"source {name!r}: `require_all_entities` names "
                    f"{', '.join(named)} — one distinct entity type, so there is nothing to "
                    "AND and no predicate is rewritten; it resolves only as a one-column key",
                    path=rel,
                    line=fixed_line,
                    hint=(
                        "two or more types to get the AND, or say the one-column key as "
                        "`identity_keys: [[type]]` so it reads as the claim it is"
                    ),
                )
            )
        _check_key_members_bound(
            named,
            "`require_all_entities`",
            name=name,
            bound=bound,
            rel=rel,
            line=fixed_line,
            diags=diags,
        )


def _check_key_members_bound(
    members: List[str],
    where: str,
    *,
    name: str,
    bound: set,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Warn on a key member this source declares no `entity_bindings` for.

    A key is satisfied wholly or not at all, so one unbound member disqualifies the whole
    candidate silently on every incident. A warning rather than an error because
    ``entity_bindings`` is not the only route: ``map_entities`` also binds off the discovered
    schema, so a member named nowhere here can still resolve at run time.
    """
    if not bound:
        # Nothing declared at all: the source relies on discovery for every type, and this
        # check would then fire on all of them without knowing anything.
        return
    unbound = [m for m in members if m not in bound]
    if not unbound:
        return
    diags.append(
        _diag(
            "warning",
            "actor-key-member-not-bound",
            f"source {name!r}: {where} names {', '.join(unbound)}, which this source "
            "declares no `entity_bindings` for — the whole candidate is skipped unless the "
            "mapper happens to bind the type off the discovered schema",
            path=rel,
            line=line,
            hint="add the column(s) under `entity_bindings`, or drop the member",
        )
    )


def _report_unread_keys(
    doc: Dict[str, Any], rel: str, diags: List[Dict[str, Any]], *, where: str
) -> None:
    for key in sorted(doc):
        if not _is_read_by_engine(str(key)):
            diags.append(
                _diag(
                    "warning",
                    "unread-pack-key",
                    f"{where} declares {key!r} and nothing in the engine reads it, so it "
                    "does nothing",
                    path=rel,
                    hint="wire it, rename it to the key the engine reads, or delete it",
                )
            )


def _check_data_stems(
    root: Path, diags: List[Dict[str, Any]], counts: Dict[str, int]
) -> None:
    """Reference data is keyed by file stem; two stems means one file silently lost.

    Two severities: root-vs-use-case collision is documented precedence (the use case wins),
    worth noting but not a defect. Two use cases colliding has no declared winner; the loader
    takes the last one sorted alphabetically.
    """
    by_stem: Dict[str, List[str]] = {}
    for data_dir in [root / "data"] + sorted((root / "use_cases").glob("*/data")):
        if not data_dir.is_dir():
            continue
        for path in sorted(data_dir.glob("*.yaml")) + sorted(data_dir.glob("*.yml")):
            rel = str(PurePosixPath(path.relative_to(root)))
            by_stem.setdefault(path.stem, []).append(rel)
    counts["data_files"] = sum(len(v) for v in by_stem.values())
    for stem, paths in sorted(by_stem.items()):
        if len(paths) < 2:
            continue
        in_use_cases = [p for p in paths if p.startswith("use_cases/")]
        if len(in_use_cases) > 1:
            diags.append(
                _diag(
                    "error",
                    "duplicate-data-stem",
                    f"{len(in_use_cases)} use cases each ship a data file named "
                    f"{stem!r}; only the last one loaded survives",
                    path=in_use_cases[-1],
                    detail="; ".join(in_use_cases),
                    hint="rename one — the stem IS the key the engine looks it up by",
                )
            )
        else:
            diags.append(
                _diag(
                    "warning",
                    "duplicate-data-stem",
                    f"{stem!r} exists at the pack root and inside a use case; the use "
                    "case's copy wins",
                    path=paths[-1],
                    detail="; ".join(paths),
                    hint="intended precedence, or a leftover copy of the root file",
                )
            )


def _check_correlation_specs(
    root: Path, diags: List[Dict[str, Any]], counts: Dict[str, int]
) -> None:
    """Warn when a correlation spec's discriminating tokens are short and owned by it alone.

    A misselected procedure produces a confident wrong verdict. The scorer weights each token
    by inverse spec frequency, so a unique token carries maximum weight; matched as substrings,
    a short unique token fires inside unrelated words. Warning because a short token may be
    exactly the right discriminator in some domains. Reported per spec with the tokens named.
    Reads frontmatter directly to avoid treating a parse error as a spec declaring no block.
    """
    specs: List[Tuple[str, str, Any]] = []  # (rel path, title, keys)
    for path in sorted(root.rglob("playbooks/*.md")):
        rel = str(PurePosixPath(path.relative_to(root)))
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        front: Any = {}
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                try:
                    front = yaml.safe_load(text[3:end])
                except yaml.YAMLError:
                    front = {}  # already reported by `_check_frontmatter`
        front = front if isinstance(front, dict) else {}
        block = front.get("correlation")
        if not isinstance(block, dict):
            continue
        # The loader's own title rule: the frontmatter title, else the file stem.
        title = str(front.get("title", "") or path.stem)
        specs.append((rel, title, block.get("keys")))
    counts["correlation_specs"] = len(specs)
    # With one spec the score is moot: the engine returns the only candidate regardless, so a
    # lone title's vocabulary discriminates nothing and reporting it would be noise.
    if len(specs) < 2:
        return

    frequency: Dict[str, int] = {}
    tokens_by_spec: List[Set[str]] = []
    for _rel, title, keys in specs:
        tokens = _spec_tokens(title, keys)
        tokens_by_spec.append(tokens)
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1

    for (rel, title, _keys), tokens in zip(specs, tokens_by_spec):
        weak = sorted(
            t
            for t in tokens
            if frequency.get(t) == 1 and (t in _STOPWORDS or len(t) <= 2)
        )
        if not weak:
            continue
        diags.append(
            _diag(
                "warning",
                "spec-title-weak-discriminator",
                f"{title}: {len(weak)} token(s) of this correlation spec's vocabulary are "
                "owned by it alone, so the selector weights each at its maximum, but they "
                "are function words or two characters long and are matched as substrings",
                path=rel,
                detail=", ".join(weak),
                hint="lengthen the title with words the other procedures do not use, or "
                "accept it if the short token really is this procedure's discriminator",
            )
        )


def _check_use_case_shape(
    root: Path, diags: List[Dict[str, Any]], counts: Dict[str, int]
) -> None:
    use_cases = root / "use_cases"
    n_uc = 0
    if use_cases.is_dir():
        for uc_dir in sorted(p for p in use_cases.iterdir() if p.is_dir()):
            n_uc += 1
            if (uc_dir / "rules.yaml").is_file():
                continue
            diags.append(
                _diag(
                    "info",
                    "playbook-only-use-case",
                    f"{uc_dir.name}: narrative only — it has no rules.yaml, so no "
                    "deterministic verdict is reached for it",
                    path=f"use_cases/{uc_dir.name}",
                    hint="the ordinary shape for a use case that is documented, not automated",
                )
            )
    counts["use_cases"] = n_uc
    counts["playbooks"] = len(list(root.rglob("playbooks/*.md")))
    counts["concepts"] = len(list(root.rglob("concepts/*.md")))
    counts["cases"] = len(list(root.rglob("cases/*.md")))
    counts["schemas"] = len(list((root / "schemas").glob("*.yaml"))) + len(
        list((root / "schemas").glob("*.yml"))
    )


#: Projection operations that can SHORTEN a value, and so can collapse two unrelated ones onto
#: one key. A form built only from the rest cannot, which is why the ``min_length`` warning is
#: conditional: a fixed-width code folded to upper case needs no floor.
_SHORTENING_OPS = ("keep", "strip", "prefix", "suffix", "tokens", "collapse_repeats")

#: Top-level keys an equivalence form may declare. Checked because a misspelling is silent in
#: the worst direction: a typo'd ``compare`` leaves a projection-only form, which is a DIFFERENT
#: relation that still evaluates and still reports the form's name.
_FORM_KEYS = ("project", "compare", "linkage", "min_length", "description")


def _form_op(entry: Any) -> Tuple[str, Any, str]:
    """One pipeline step -> ``(op, param, problem)``; ``problem`` is empty when it is readable."""
    if not isinstance(entry, dict):
        return "", None, "each step is a mapping of one operation to its parameter"
    keys = [str(k) for k in entry]
    if len(keys) != 1:
        return (
            "",
            None,
            "each step declares exactly one operation, and this one declares "
            + (", ".join(sorted(keys)) if keys else "none")
            + " — two operations in one step is an order nobody authored",
        )
    return keys[0], entry[keys[0]], ""


def _check_form_step(
    op: str,
    param: Any,
    *,
    family: str,
    vocabulary: Tuple[str, ...],
    form_name: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> bool:
    """Check one operation and its parameter; ``True`` when the step can only yield ``""``.

    Every failure here is an ERROR rather than a warning for one reason: an operation the engine
    does not apply, or applies with a parameter it cannot read, is a relation OTHER than the one
    declared — and the condition still evaluates, still names the form, and reports a number
    arrived at some other way.
    """
    if vocabulary and op not in vocabulary:
        diags.append(
            _diag(
                "error",
                "equivalence-form-unknown-op",
                f"form {form_name!r}: {op!r} is not one of the {family} operations, so the "
                "relation this form declares is not the one the engine would apply",
                path=rel,
                line=line,
                detail=f"available: {', '.join(vocabulary)}",
            )
        )
        return False
    required = dict(
        zip(
            _engine_mapping_keys("_OP_PARAMS"),
            _op_param_descriptions(),
        )
    ).get(op, "")
    ints = (
        "prefix",
        "suffix",
        "shared_prefix",
        "shared_suffix",
        "edit_distance",
        "min_overlap",
    )
    flags = ("collapse_repeats", "exact", "contains")
    bad = ""
    if op in flags:
        if param is not True:
            bad = "true"
    elif op in ints:
        if isinstance(param, bool) or not isinstance(param, int):
            # `prefix: true` is a declaration that lost its number, and `int(True)` would
            # silently make it 1 — a one-character key that collapses nearly everything.
            bad = "a whole number"
    elif op == "case":
        if str(param or "") not in ("upper", "lower", "fold"):
            bad = "upper, lower or fold"
    elif op == "keep":
        if str(param or "") not in ("alnum", "alpha", "digits"):
            bad = "alnum, alpha or digits"
    elif op == "strip":
        if not isinstance(param, str) or not param:
            bad = "the characters to remove from both ends"
    elif op == "map":
        if not isinstance(param, dict) or not param:
            bad = "a non-empty mapping of value to canonical value"
    elif op == "tokens":
        if not isinstance(param, dict) or not str(param.get("split", "") or ""):
            bad = "a mapping declaring `split`"
        else:
            order = str(param.get("order", "") or "")
            orders = _engine_literals("_TOKEN_ORDERS")
            if order and orders and order not in orders:
                diags.append(
                    _diag(
                        "error",
                        "equivalence-form-token-order",
                        f"form {form_name!r}: `tokens.order` is {order!r}, which the engine "
                        "does not apply, so the tokens keep the order they were written in",
                        path=rel,
                        line=line,
                        detail=f"available: {', '.join(orders)}",
                    )
                )
            take = param.get("take")
            if take is not None and (
                isinstance(take, bool) or not isinstance(take, int)
            ):
                bad = "a whole `take` count, or none to keep every token"
            elif isinstance(take, int) and not isinstance(take, bool) and take <= 0:
                return True
    if bad:
        diags.append(
            _diag(
                "error",
                "equivalence-form-op-parameter",
                f"form {form_name!r}: `{op}` takes {required or bad} and declares "
                f"{param!r}, so the engine cannot apply it and every reading under this "
                "form is refused",
                path=rel,
                line=line,
                hint=f"declare `{op}: <{bad}>`",
            )
        )
        return False
    if op in ("prefix", "suffix") and isinstance(param, int) and param <= 0:
        return True
    return False


def _op_param_descriptions() -> Tuple[str, ...]:
    """The engine's own wording for what each operation requires, in declaration order.

    Read from ``_OP_PARAMS``' values rather than restated, so the message an author sees is the
    sentence the engine's own vocabulary block carries.
    """
    if "op_param_values" not in _CORPUS_CACHE:
        try:
            text = (REPO_ROOT / "src" / "correlation.py").read_text(encoding="utf-8")
        except OSError:
            text = ""
        m = re.search(r"\n_OP_PARAMS\s*=\s*\{([^}]*)\}", text)
        found = tuple(re.findall(r':\s*"([^"]+)"', m.group(1))) if m else ()
        _CORPUS_CACHE["op_param_values"] = found
    return _CORPUS_CACHE["op_param_values"]


def _check_equivalence_forms(
    root: Path,
    yaml_docs: Dict[str, Any],
    diags: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """Validate ``shared/equivalence_forms.yaml`` and return the forms it declares.

    A form declares what "the same thing" means for this deployment, so getting it wrong does
    not error at run time — it changes the relation and reports the number under the form's own
    name. The one warning is ``min_length``: a form that can shorten a value and declares no
    floor collapses unrelated values onto one key, which fabricates a finding rather than
    losing one, but a fixed-width code legitimately needs no floor.
    """
    rel = "shared/equivalence_forms.yaml"
    if not (root / rel).is_file():
        return {}
    forms = _read_equivalence_forms(root / rel)
    counts["equivalence_forms"] = len(forms)
    text = (root / rel).read_text(encoding="utf-8", errors="replace")
    # The loader skips a malformed form with a log line, which is the one place a form can
    # disappear silently: a condition then names a form no pack declares and is refused, with
    # the reason in a log nobody is reading. Reported against the parsed document, not the
    # loader's output, so what was dropped is named.
    doc = yaml_docs.get(rel)
    if doc is not None and not isinstance(doc, dict):
        diags.append(
            _diag(
                "error",
                "equivalence-forms-not-a-mapping",
                "shared/equivalence_forms.yaml is a flat map of form name to pipeline, and "
                f"this parses to {type(doc).__name__} — no form is declared at all",
                path=rel,
                line=1,
            )
        )
    for name in sorted(k for k in (doc or {}) if isinstance(doc, dict)):
        if str(name) not in forms:
            diags.append(
                _diag(
                    "error",
                    "equivalence-form-not-a-mapping",
                    f"form {str(name)!r} is not a mapping, so it was dropped at load — a "
                    "condition naming it is refused as a form no pack declares",
                    path=rel,
                    line=_find_key_line(text, str(name)),
                )
            )
    if not forms:
        return forms
    projection = _engine_literals("_PROJECTION_OPS")
    comparison = _engine_literals("_COMPARISON_OPS")
    linkages = _engine_literals("_LINKAGES")
    for name, form in sorted(forms.items()):
        line = _find_key_line(text, name)
        unknown = [k for k in form if str(k) not in _FORM_KEYS]
        if unknown:
            diags.append(
                _diag(
                    "warning",
                    "equivalence-form-unknown-key",
                    f"form {name!r} declares {', '.join(sorted(unknown))}, which the engine "
                    "does not read — a misspelled `compare` leaves a projection-only form, "
                    "which is a different relation that still evaluates",
                    path=rel,
                    line=line,
                    detail=f"read: {', '.join(_FORM_KEYS)}",
                )
            )
        steps = form.get("project")
        always_empty = False
        shortens = False
        if steps is not None and not isinstance(steps, list):
            diags.append(
                _diag(
                    "error",
                    "equivalence-form-project-shape",
                    f"form {name!r}: `project` is an ordered list of single-operation steps, "
                    f"and this declares {type(steps).__name__}",
                    path=rel,
                    line=line,
                    hint="the order is the semantics: `case` before `map` folds the keys first",
                )
            )
            steps = []
        for entry in steps or []:
            op, param, problem = _form_op(entry)
            if problem:
                diags.append(
                    _diag(
                        "error",
                        "equivalence-form-step-shape",
                        f"form {name!r}: {problem}",
                        path=rel,
                        line=line,
                    )
                )
                continue
            shortens = shortens or op in _SHORTENING_OPS
            always_empty = always_empty or _check_form_step(
                op,
                param,
                family="projection",
                vocabulary=projection,
                form_name=name,
                rel=rel,
                line=line,
                diags=diags,
            )
        if always_empty:
            # Every value reduces to the empty string, which is unresolvable, so the form
            # reads nothing at all — and a check whose values are all unresolvable reports a
            # count of zero that looks exactly like a clean population.
            diags.append(
                _diag(
                    "error",
                    "equivalence-form-always-empty",
                    f"form {name!r} can only ever produce the empty string, so every value "
                    "is unresolvable under it and no two are ever equivalent",
                    path=rel,
                    line=line,
                    hint="a `prefix`/`suffix`/`take` of zero or less keeps no characters",
                )
            )
        compare = form.get("compare")
        if compare is not None:
            op, param, problem = _form_op(compare)
            if problem:
                diags.append(
                    _diag(
                        "error",
                        "equivalence-form-compare-shape",
                        f"form {name!r}: `compare` is one operation — {problem}",
                        path=rel,
                        line=line,
                    )
                )
            else:
                _check_form_step(
                    op,
                    param,
                    family="comparison",
                    vocabulary=comparison,
                    form_name=name,
                    rel=rel,
                    line=line,
                    diags=diags,
                )
        linkage = str(form.get("linkage", "") or "")
        if linkage and linkages and linkage not in linkages:
            diags.append(
                _diag(
                    "error",
                    "equivalence-form-linkage",
                    f"form {name!r}: `linkage` is {linkage!r}, which the engine does not "
                    "implement, so an unanchored comparison under this form is refused",
                    path=rel,
                    line=line,
                    detail=f"available: {', '.join(linkages)}",
                )
            )
        if linkage and compare is None:
            diags.append(
                _diag(
                    "warning",
                    "equivalence-form-linkage-noop",
                    f"form {name!r} declares `linkage` and no `compare` — a projection is "
                    "transitive, so its classes are the same under either linkage and the "
                    "key changes nothing",
                    path=rel,
                    line=line,
                )
            )
        floor = form.get("min_length")
        if floor is not None and (
            isinstance(floor, bool) or not isinstance(floor, int)
        ):
            diags.append(
                _diag(
                    "error",
                    "equivalence-form-min-length",
                    f"form {name!r}: `min_length` is {floor!r} and must be a whole number — "
                    "the engine cannot read it, so the floor it declares is not applied",
                    path=rel,
                    line=line,
                )
            )
        elif floor is None and shortens:
            diags.append(
                _diag(
                    "warning",
                    "equivalence-form-no-min-length",
                    f"form {name!r} shortens the values it reads and declares no "
                    "`min_length`, so a value too short for the pipeline collapses onto "
                    "every other short one",
                    path=rel,
                    line=line,
                    hint=(
                        "declare `min_length:` at the shortest key that still identifies "
                        "one thing; a fixed-width code needs none"
                    ),
                )
            )
    return forms


def _check_rulesets(
    root: Path,
    yaml_docs: Dict[str, Any],
    diags: List[Dict[str, Any]],
    counts: Dict[str, int],
    forms: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Every ruleset, with its shared-check imports resolved first.

    Resolved first because that is what the evaluator sees: a ``kind`` or a ``source`` a
    condition never mentions may arrive from the library entry it imports, and checking the
    raw text would report those as missing while checking the resolved form reports what
    actually runs.
    """
    library = _read_shared_checks(root / "shared" / "checks")
    counts["shared_checks"] = len(library)
    catalog = yaml_docs.get("source_catalog.yaml")
    physical = {
        str(s.get("name", "") or "")
        for s in ((catalog or {}).get("sources") or [])
        if isinstance(s, dict)
    }
    pack_data = _read_data_dir(root / "data")
    for data_dir in sorted((root / "use_cases").glob("*/data")):
        pack_data.update(_read_data_dir(data_dir))
    decoded = _decoded_table_paths(catalog)
    projected = _projected_paths(catalog)
    renames = _projection_renames(catalog)

    n_rulesets = 0
    n_conditions = 0
    n_path_lists = 0
    n_projected_lists = 0
    files = [("rulesets.yaml", None)]
    if (root / "use_cases").is_dir():
        files += [
            (f"use_cases/{d.name}/rules.yaml", d.name)
            for d in sorted(p for p in (root / "use_cases").iterdir() if p.is_dir())
        ]
    declared_keys: Set[str] = set()
    #: Kept per ruleset for the pack-level link-graph reconciliation below, which is the one
    #: check that cannot be made from inside a single ruleset: an inbound signal is read on
    #: another procedure's rows, so who retrieves the source is not knowable here.
    resolved: Dict[str, Dict[str, Any]] = {}
    #: Where each resolved ruleset was read from, for line-accurate reporting. Kept beside
    #: ``resolved`` because the escalation check names another ruleset and the complete key
    #: set is not known until the loop has finished.
    origins: Dict[str, Tuple[str, str]] = {}
    n_entry_signals = 0
    for rel, use_case in files:
        doc = yaml_docs.get(rel)
        if not isinstance(doc, dict):
            continue
        verdicts = doc.get("verdicts")
        if not isinstance(verdicts, dict):
            continue
        declared_keys |= {str(k) for k in verdicts}
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for key, raw in verdicts.items():
            n_rulesets += 1
            if not isinstance(raw, dict):
                diags.append(
                    _diag(
                        "error",
                        "ruleset-not-a-mapping",
                        f"ruleset {key!r} is not a mapping",
                        path=rel,
                        line=_find_key_line(text, str(key)),
                    )
                )
                continue
            try:
                spec = _resolve_check_imports(raw, library, str(key))
            except ValueError as exc:
                diags.append(
                    _diag(
                        "error",
                        "unresolvable-check-import",
                        f"ruleset {key!r} imports a shared check that does not exist, "
                        "which stops the pack loading",
                        path=rel,
                        line=_find_key_line(text, "use"),
                        detail=str(exc),
                        hint=(
                            "the only deliberately fatal path in pack loading — a dropped "
                            "condition would otherwise read exactly like a source that "
                            "returned no rows"
                        ),
                    )
                )
                continue
            n_conditions += len(spec.get("conditions") or [])
            resolved[str(key)] = spec
            origins[str(key)] = (rel, text)
            signals = spec.get("entry_signals")
            n_entry_signals += len(signals) if isinstance(signals, list) else 0
            checked_lists, checked_projected = _check_one_ruleset(
                spec,
                raw,
                key=str(key),
                rel=rel,
                text=text,
                use_case=use_case,
                root=root,
                physical=physical,
                pack_data=pack_data,
                forms=forms or {},
                decoded=decoded,
                projected=projected,
                renames=renames,
                diags=diags,
            )
            n_path_lists += checked_lists
            n_projected_lists += checked_projected
    counts["rulesets"] = n_rulesets
    counts["conditions"] = n_conditions
    counts["field_path_lists"] = n_path_lists
    counts["projected_path_lists"] = n_projected_lists
    counts["entry_signals"] = n_entry_signals
    # After the loop, with every key known: an escalation override names a source procedure,
    # and a ruleset declared further down in the same file is a valid name.
    for rkey, spec in resolved.items():
        rel, text = origins.get(rkey, ("rulesets.yaml", ""))
        _check_link_escalation(
            spec,
            key=rkey,
            rel=rel,
            text=text,
            ruleset_keys=set(resolved),
            diags=diags,
        )
    _check_default_ruleset(root, yaml_docs, declared_keys, diags)
    _check_link_graph(root, catalog, resolved, diags, counts)


def _check_default_ruleset(
    root: Path,
    yaml_docs: Dict[str, Any],
    declared_keys: Set[str],
    diags: List[Dict[str, Any]],
) -> None:
    """``default_ruleset`` must name an existing ruleset and is read from one place only.

    Error on both halves. A name matching nothing falls back to first-declared (use-case
    directory order), so the pack appears to have taken the decision while it has not.
    A ``default_ruleset:`` inside a use case's ``rules.yaml`` is inert: the loader reads
    it off the flat-root file only.
    """
    for rel in sorted(yaml_docs):
        if rel != "rulesets.yaml" and not re.fullmatch(
            r"use_cases/[^/]+/rules\.yaml", rel
        ):
            continue
        doc = yaml_docs.get(rel)
        if not isinstance(doc, dict):
            continue
        declared = str(doc.get("default_ruleset", "") or "").strip()
        if not declared:
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        line = _find_key_line(text, "default_ruleset")
        if rel != "rulesets.yaml":
            diags.append(
                _diag(
                    "error",
                    "default-ruleset-not-at-pack-root",
                    "default_ruleset is only read from the pack root's rulesets.yaml, so "
                    f"declaring it in {rel} does nothing",
                    path=rel,
                    line=line,
                    hint=(
                        "move it to rulesets.yaml at the pack root, beside (or instead of) "
                        "any flat-root verdicts"
                    ),
                )
            )
            continue
        if declared not in declared_keys:
            diags.append(
                _diag(
                    "error",
                    "default-ruleset-unknown",
                    f"default_ruleset names {declared!r}, which no ruleset declares",
                    path=rel,
                    line=line,
                    detail="rulesets declared: "
                    + (", ".join(sorted(declared_keys)) or "none"),
                    hint=(
                        "the fallback then reverts to the first declared ruleset, which is "
                        "use-case directory order — so the default moves whenever a "
                        "procedure is added under an earlier name"
                    ),
                )
            )


def _evaluable(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The conditions that can ever resolve to ``pass`` or ``fail``.

    Only ``kind: stub`` cannot: it is a placeholder for an unconfirmed data path and returns
    ``unknown`` unconditionally. An unknown ``kind`` is not excluded here; it is already an
    error and treating it as unevaluable would let one defect mask another.
    """
    return [c for c in conditions if str(c.get("kind", "") or "") != "stub"]


def _check_one_ruleset(
    spec: Dict[str, Any],
    raw: Dict[str, Any],
    *,
    key: str,
    rel: str,
    text: str,
    use_case: Optional[str],
    root: Path,
    physical: Set[str],
    pack_data: Dict[str, Any],
    forms: Dict[str, Dict[str, Any]],
    decoded: Dict[str, List[str]],
    projected: Dict[str, List[str]],
    renames: Dict[str, Dict[str, str]],
    diags: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Every per-ruleset check.

    Returns two independent coverage counts: how many field-path lists were checked against
    a schema doc, and how many against a declared ``projection``. Two numbers because the
    two checks are silent on different sets: a source with no inventory silences both, a
    backend that ignores ``projection`` only the second.
    """
    conditions = [c for c in (spec.get("conditions") or []) if isinstance(c, dict)]
    _check_indicator_threshold(spec, key=key, rel=rel, text=text, diags=diags)
    if not conditions:
        diags.append(
            _diag(
                "warning",
                "ruleset-no-conditions",
                f"ruleset {key!r} has no conditions, so it reaches no verdict on evidence",
                path=rel,
                line=_find_key_line(text, key),
            )
        )

    # --- keys nothing reads ---------------------------------------------------
    # Scoped to top level, conditions, and groups. One level down the maps are keyed by
    # author vocabulary, and a check there would need an exemption list to be useful.
    scanned: Set[str] = set(spec)
    for cond in conditions:
        scanned |= set(cond)
    for group in spec.get("condition_groups") or []:
        if isinstance(group, dict):
            scanned |= set(group)
    for name in sorted(scanned):
        if name == "use":
            continue  # resolved away by the importer; never reaches the engine
        if not _is_read_by_engine(str(name)):
            diags.append(
                _diag(
                    "warning",
                    "unread-pack-key",
                    f"ruleset {key!r} declares {name!r} and nothing in the engine reads "
                    "it, so it does nothing",
                    path=rel,
                    line=_find_key_line(text, str(name)),
                    hint="wire it, rename it to the key the engine reads, or delete it",
                )
            )

    # --- sources --------------------------------------------------------------
    declared = spec.get("sources") or {}
    if isinstance(declared, dict):
        for logical, target in sorted(declared.items()):
            if str(target) not in physical:
                diags.append(
                    _diag(
                        "error",
                        "unknown-physical-source",
                        f"{logical!r} points at {target!r}, which the source catalog does "
                        "not declare — it can never be retrieved",
                        path=rel,
                        line=_find_key_line(text, str(logical)),
                        hint="add it to source_catalog.yaml, or point at a source that is there",
                    )
                )
    used: Set[str] = set()
    # `entry_signals` is excluded: a signal naming a source is not a condition reading it.
    # Walking it would report a sibling procedure's name as `unknown-ruleset-source` and
    # suppress `orphan-logical-source`. Its own check is `entry-signal-unknown-source`.
    _walk_source_refs(
        {k: v for k, v in spec.items() if k not in ("sources", "entry_signals")}, used
    )
    if isinstance(declared, dict):
        for logical in sorted(used - set(declared) - physical):
            diags.append(
                _diag(
                    "error",
                    "unknown-ruleset-source",
                    f"ruleset {key!r} reads a source named {logical!r} that it never "
                    "declares, so every check on it is `unknown`",
                    path=rel,
                    line=_find_key_line(text, "sources"),
                    hint="add it to this ruleset's `sources:` map",
                )
            )
        # An unused declared source is the other direction and is a warning, not an error:
        # a `sources:` entry is a hard dependency on every run, and one no condition reads
        # spends its full scan cost for nothing. The pack still works.
        for logical in sorted(set(declared) - used):
            diags.append(
                _diag(
                    "warning",
                    "orphan-logical-source",
                    f"{logical!r} is retrieved on every run and no condition reads it",
                    path=rel,
                    line=_find_key_line(text, str(logical)),
                    hint=(
                        "if relevance depends on the incident, drop it here and let the "
                        "source's own selection_guidance decide instead"
                    ),
                )
            )

    # --- as_of ----------------------------------------------------------------
    # Every way to get `as_of` wrong is silent: a misnamed source is skipped (the log is
    # adjudicated whole), and a missing key makes the entry a no-op with no note. All
    # three keys are required together; `_as_of_rows` abstains on any half-written entry.
    as_of = spec.get("as_of") or {}
    if isinstance(as_of, dict):
        for logical, entry in sorted(as_of.items()):
            if str(logical) not in declared:
                diags.append(
                    _diag(
                        "error",
                        "as-of-unknown-source",
                        f"as_of declares {logical!r}, which this ruleset's `sources:` map "
                        "does not — so nothing is restricted to the incident instant and "
                        "changes made after the alert are adjudicated as evidence",
                        path=rel,
                        line=_find_key_line(text, str(logical)),
                        hint="use the LOGICAL source name as spelled in `sources:`",
                    )
                )
                continue
            entry = entry if isinstance(entry, dict) else {}

            def _listed(key: str) -> bool:
                return bool([v for v in (entry.get(key) or []) if str(v).strip()])

            if not _listed("timestamp_fields"):
                diags.append(
                    _diag(
                        "error",
                        "as-of-no-timestamp",
                        f"as_of.{logical} declares no `timestamp_fields`, so the engine "
                        "cannot tell which version was written when and the whole chain is "
                        "adjudicated — the entry does nothing",
                        path=rel,
                        line=_find_key_line(text, str(logical)),
                        hint=(
                            "list the column holding each version's WRITE time, "
                            "sub-day granularity first"
                        ),
                    )
                )
            # The actor half. Its absence is the more dangerous omission of the two, because a
            # reader of the YAML sees a timestamp and a source and concludes the rule is armed.
            if not _listed("actor_fields"):
                diags.append(
                    _diag(
                        "error",
                        "as-of-no-actor-field",
                        f"as_of.{logical} declares no `actor_fields`, so the engine cannot "
                        "tell the responder's later changes from the subject's own and the "
                        "entry does nothing — a boundary on time alone would discard the "
                        "subject's later conduct, including anything exculpatory",
                        path=rel,
                        line=_find_key_line(text, str(logical)),
                        hint=(
                            "list the column holding the identity that WROTE each version "
                            "(the last-updator, not the record's creator)"
                        ),
                    )
                )
            if not _listed("actor_entities"):
                diags.append(
                    _diag(
                        "error",
                        "as-of-no-actor-entity",
                        f"as_of.{logical} declares no `actor_entities`, so the engine has no "
                        "identity from the incident to compare a writer against and the entry "
                        "does nothing",
                        path=rel,
                        line=_find_key_line(text, str(logical)),
                        hint=(
                            "name the entity type(s) whose values are acting identities, "
                            "as extracted from the incident"
                        ),
                    )
                )

    # --- follow-up passes -----------------------------------------------------
    _check_follow_up_passes(
        spec,
        key=key,
        rel=rel,
        text=text,
        declared=declared if isinstance(declared, dict) else {},
        diags=diags,
    )

    # --- inbound cross-procedure signals --------------------------------------
    _check_entry_signals(
        spec,
        key=key,
        rel=rel,
        text=text,
        declared=declared if isinstance(declared, dict) else {},
        physical=physical,
        diags=diags,
    )

    # --- conditions -----------------------------------------------------------
    kinds = condition_kinds()
    honours_expected_label = expected_label_kinds()
    groups = {
        str(g.get("id", "") or "")
        for g in (spec.get("condition_groups") or [])
        if isinstance(g, dict)
    }
    seen_ids: Dict[str, int] = {}
    has_scope_gate = False
    raw_by_index = [c for c in (raw.get("conditions") or []) if isinstance(c, dict)]
    for index, cond in enumerate(conditions):
        cid = str(cond.get("id", "") or "")
        line = _find_id_line(text, cid)
        if cid:
            seen_ids[cid] = seen_ids.get(cid, 0) + 1
        else:
            diags.append(
                _diag(
                    "warning",
                    "condition-missing-id",
                    f"a condition in {key!r} has no `id`, so it is named after its kind "
                    "and collides with every other condition of that kind",
                    path=rel,
                    detail=f"kind: {cond.get('kind', '?')}",
                )
            )
        kind = str(cond.get("kind", "") or "")
        if kinds and kind not in kinds:
            diags.append(
                _diag(
                    "error",
                    "unknown-condition-kind",
                    f"{cid or '(unnamed)'}: kind {kind or '(missing)'!r} is not one the "
                    "evaluator dispatches, so this check is never evaluated",
                    path=rel,
                    line=line,
                    detail=f"available: {', '.join(sorted(kinds))}",
                    hint=(
                        "an unevaluated check reads in the report exactly like one whose "
                        "source returned no rows"
                    ),
                )
            )
        if cond.get("expected_label") and kind not in honours_expected_label:
            diags.append(
                _diag(
                    "warning",
                    "expected-label-noop",
                    f"{cid or '(unnamed)'}: `expected_label` is a no-op on kind "
                    f"{kind!r} — that wording is never printed",
                    path=rel,
                    line=line,
                    detail=(
                        "honoured only by: " + ", ".join(sorted(honours_expected_label))
                    ),
                    hint="put the wording in `pass_detail` / `fail_detail` instead",
                )
            )
        # `inconclusive_blocks_pass` is a no-op without `inconclusive_patterns`: nothing is
        # ever dropped, so the check clears as if the key were absent. The two spellings are
        # one word apart and the failure is a silent clear.
        if cond.get("inconclusive_blocks_pass") and not cond.get(
            "inconclusive_patterns"
        ):
            diags.append(
                _diag(
                    "warning",
                    "inconclusive-blocks-pass-noop",
                    f"{cid or '(unnamed)'}: `inconclusive_blocks_pass` is declared with no "
                    "`inconclusive_patterns`, so no value is ever dropped and the key "
                    "changes nothing",
                    path=rel,
                    line=line,
                    hint=(
                        "declare the codes whose meaning is unestablished, or drop the key "
                        "— a clear that this was meant to withhold is still reported"
                    ),
                )
            )
        # `ordinary_patterns` closes the vocabulary: a value outside all three lists is
        # withheld rather than read as a non-match. Both silent failures below let the check
        # return the answer the author declared this key to prevent.
        ordinary_kinds = kinds_reading("ordinary_patterns")
        if cond.get("ordinary_patterns") and ordinary_kinds and kind not in ordinary_kinds:
            diags.append(
                _diag(
                    "warning",
                    "ordinary-patterns-noop",
                    f"{cid or '(unnamed)'}: `ordinary_patterns` is a no-op on kind "
                    f"{kind!r} — the vocabulary is not closed and an unlisted value still "
                    "reads as an ordinary non-match",
                    path=rel,
                    line=line,
                    detail="read only by: " + ", ".join(sorted(ordinary_kinds)),
                    hint=(
                        "an unclassified value produces the same finding as a classified "
                        "one, which is what this key exists to prevent"
                    ),
                )
            )
        # The same string in two classification lists contradicts the pack; which list wins
        # follows the evaluator's filter order. Checked on the string only: regex overlap is
        # not decidable here.
        _list_keys = ("patterns", "inconclusive_patterns", "ordinary_patterns")
        _seen: Dict[str, str] = {}
        for _key in _list_keys:
            _decl = cond.get(_key) or []
            if isinstance(_decl, str):
                _decl = [_decl]
            for _pat in _decl:
                _pat = str(_pat)
                if _pat in _seen and _seen[_pat] != _key:
                    diags.append(
                        _diag(
                            "warning",
                            "pattern-in-two-classifications",
                            f"{cid or '(unnamed)'}: pattern {_pat!r} is declared in both "
                            f"`{_seen[_pat]}` and `{_key}`, so this check classifies the same "
                            "value two ways",
                            path=rel,
                            line=line,
                            hint=(
                                "which one applies follows the evaluator's filter order, not "
                                "the declaration — put the value in exactly one list"
                            ),
                        )
                    )
                else:
                    _seen.setdefault(_pat, _key)
        _check_pair_by(cond, cid, kind, rel, line, diags)
        _check_subject_scope(cond, cid, spec, rel, line, diags)
        _check_cohort_subject_rows(cond, cid, kind, rel, line, diags)
        _check_row_match_scope(cond, cid, rel, line, diags)
        _check_numeric_compare(cond, cid, kind, rel, line, diags)
        _check_baseline(cond, cid, kind, rel, line, diags)
        _check_event_order(cond, cid, kind, rel, line, diags)
        _check_value_equivalence(cond, cid, kind, rel, line, diags)
        _check_condition_form(cond, cid, kind, forms, rel, line, diags)
        _check_composite(cond, cid, kind, forms, rel, line, diags)
        group = str(cond.get("report_group", "") or "")
        if group and groups and group not in groups:
            diags.append(
                _diag(
                    "warning",
                    "unknown-report-group",
                    f"{cid or '(unnamed)'}: report_group {group!r} is not one this "
                    "ruleset declares, so the report has no heading to file it under",
                    path=rel,
                    line=line,
                    detail=f"declared: {', '.join(sorted(groups))}",
                )
            )
        elif not group and groups:
            # A blank `report_group` is the default, so this is the most common form.
            # Separate from the check above: a missing name is a typo or a deleted group,
            # while a blank is a condition that was never filed.
            diags.append(
                _diag(
                    "warning",
                    "missing-report-group",
                    f"{cid or '(unnamed)'}: this ruleset declares reporting groups and "
                    "this condition names none, so it prints outside all of them",
                    path=rel,
                    line=line,
                    detail=f"declared: {', '.join(sorted(groups))}",
                    hint="add `report_group:` naming one of the declared groups",
                )
            )
        if str(cond.get("gate", "") or "") == "scope":
            has_scope_gate = True
        lookup = cond.get("lookup")
        if isinstance(lookup, dict) and lookup.get("data"):
            stem = str(lookup["data"])
            if stem not in pack_data:
                diags.append(
                    _diag(
                        "error",
                        "unknown-lookup-data",
                        f"{cid or '(unnamed)'}: looks up {stem!r}, and no data/ file has "
                        "that stem — the check can only ever be `unknown`",
                        path=rel,
                        line=line,
                        detail=f"available: {', '.join(sorted(pack_data)) or '(none)'}",
                    )
                )
        _check_condition_bound(cond, cid=cid, rel=rel, line=line, diags=diags)
        if str(cond.get("polarity", "") or "") == "fraud_indicator":
            _check_indicator_label(
                cond,
                raw_by_index[index] if index < len(raw_by_index) else {},
                cid=cid,
                rel=rel,
                line=line,
                diags=diags,
            )
    for cid, n in sorted(seen_ids.items()):
        if n > 1:
            diags.append(
                _diag(
                    "error",
                    "duplicate-condition-id",
                    f"{cid!r} is declared {n} times in {key!r}; the report and the health "
                    "scorer both key on the id, so one of them wins arbitrarily",
                    path=rel,
                    line=_find_id_line(text, cid),
                )
            )

    labels = spec.get("labels") or {}
    if has_scope_gate and not (isinstance(labels, dict) and labels.get("out_of_scope")):
        diags.append(
            _diag(
                "warning",
                "missing-out-of-scope-label",
                f"ruleset {key!r} has a `gate: scope` condition but declares no "
                "`labels.out_of_scope`, so a gate FAIL is reported as an adjudication",
                path=rel,
                line=_find_key_line(text, "labels"),
                hint=(
                    "out of scope means the procedure does not apply, which is not the "
                    "same as a subject examined and cleared"
                ),
            )
        )

    # --- the exit taken when nothing decisive fired ---------------------------
    # `no_exclusion_fired` names the exit when every exclusion passed and no indicator
    # reached threshold. An unknown value silently falls back to `fraud`.
    exit_key = spec.get("no_exclusion_fired")
    if exit_key is not None:
        exit_name = str(exit_key).strip().lower()
        allowed = ("fraud", "false_positive", "insufficient")
        if exit_name not in allowed:
            diags.append(
                _diag(
                    "error",
                    "unknown-no-exclusion-fired-exit",
                    f"ruleset {key!r} declares `no_exclusion_fired: {exit_key!r}`, which is "
                    f"not one of {', '.join(allowed)} — the engine would fall back to "
                    "`fraud`, i.e. to the exit this declaration exists to change",
                    path=rel,
                    line=_find_key_line(text, "no_exclusion_fired"),
                    hint=f"one of {', '.join(allowed)}",
                )
            )
        elif not (isinstance(labels, dict) and str(labels.get(exit_name, "")).strip()):
            diags.append(
                _diag(
                    "warning",
                    "no-exclusion-fired-label-undeclared",
                    f"ruleset {key!r} exits to {exit_name!r} when nothing decisive fires but "
                    f"declares no `labels.{exit_name}`, so that verdict prints the engine's "
                    "generic wording rather than this procedure's",
                    path=rel,
                    line=_find_key_line(text, "no_exclusion_fired"),
                )
            )

    # --- the evidence floor guarding that exit --------------------------------
    # `min_evaluated_to_clear` is enforced only on a `false_positive` exit; any other
    # exit makes it a no-op (reported as a warning since the pack still works).
    floor_key = spec.get("min_evaluated_to_clear")
    if floor_key is not None:
        evaluable = _evaluable(conditions)
        exit_name = str(spec.get("no_exclusion_fired", "") or "").strip().lower()
        try:
            floor_val = int(floor_key)
        except (TypeError, ValueError):
            floor_val = 0
        if floor_val < 1:
            diags.append(
                _diag(
                    "error",
                    "unreadable-evidence-floor",
                    f"ruleset {key!r} declares `min_evaluated_to_clear: {floor_key!r}`, which "
                    "is not a positive integer — the engine falls back to 1, so the number "
                    "written here is not the number enforced",
                    path=rel,
                    line=_find_key_line(text, "min_evaluated_to_clear"),
                    hint=(
                        "a positive integer, at most the number of conditions that can be "
                        "evaluated (a `kind: stub` condition never can)"
                    ),
                )
            )
        elif exit_name != "false_positive":
            diags.append(
                _diag(
                    "warning",
                    "inert-evidence-floor",
                    f"ruleset {key!r} declares `min_evaluated_to_clear: {floor_val}` but its "
                    f"`no_exclusion_fired` exit is {exit_name or 'undeclared (fraud)'!r}; the "
                    "floor guards the CLEAR exit only, so it never runs here",
                    path=rel,
                    line=_find_key_line(text, "min_evaluated_to_clear"),
                )
            )
        elif floor_val > len(evaluable):
            # Checked against the evaluable count, not the declared total: a `stub`
            # condition is `unknown` by construction and can never contribute to the floor.
            # Both counts are reported so the author can decide which is the mistake.
            n_stub = len(conditions) - len(evaluable)
            declared = (
                f"declares only {len(conditions)}, of which {n_stub} declare no data path "
                f"and can never be evaluated, leaving {len(evaluable)}"
                if n_stub
                else f"declares only {len(conditions)}"
            )
            diags.append(
                _diag(
                    "warning",
                    "unreachable-evidence-floor",
                    f"ruleset {key!r} requires {floor_val} evaluated condition(s) before it "
                    f"will clear an account but {declared}, so no "
                    "subject can ever be cleared on the merits",
                    path=rel,
                    line=_find_key_line(text, "min_evaluated_to_clear"),
                )
            )

    # --- field paths against the pack's own inventory -------------------------
    # Every block naming a row path, anchored for jump-to. `sources:` is mapped logical
    # to physical (schema docs key on physical name). All top-level blocks are included
    # rather than an enumerated allowlist, which would silently skip any new block.
    blocks: List[Tuple[str, Any, int]] = [
        (
            str(cond.get("id", "") or "") or "(unnamed)",
            cond,
            _find_id_line(text, str(cond.get("id", "") or "")),
        )
        for cond in conditions
    ]
    for name, block in sorted(spec.items()):
        if name == "conditions" or name in _NON_PATH_BLOCKS:
            continue
        if isinstance(block, (dict, list)) and block:
            blocks.append((str(name), block, _find_key_line(text, str(name))))
    resolved_sources = (
        {str(k): str(v) for k, v in (declared or {}).items()}
        if isinstance(declared, dict)
        else {}
    )
    n_lists = _check_field_paths(
        blocks,
        key=key,
        rel=rel,
        root=root,
        declared=resolved_sources,
        physical=physical,
        decoded=decoded,
        renames=renames,
        diags=diags,
    )
    # A second question over the same paths: "will this leaf be in the row" (not "does it
    # exist"). Separate because the remedies are in different files. The count distinguishes
    # a silent pass (no schema doc or backend ignoring `projection`) from a checked one.
    n_projected = _check_projection_coverage(
        blocks,
        key=key,
        rel=rel,
        root=root,
        declared=resolved_sources,
        physical=physical,
        projected=projected,
        renames=renames,
        decoded=decoded,
        diags=diags,
    )

    concepts = (spec.get("case_builder") or {}).get("concepts") or []
    if concepts:
        available = _doc_ids(root / "shared" / "concepts", "concept_id")
        if use_case:
            available |= _doc_ids(
                root / "use_cases" / use_case / "concepts", "concept_id"
            )
        for name in concepts:
            if str(name) not in available:
                diags.append(
                    _diag(
                        "warning",
                        "orphan-concept-id",
                        f"the case brief asks for concept {name!r} and no concept file "
                        "provides that id, so the brief silently omits it",
                        path=rel,
                        line=_find_key_line(text, "concepts"),
                        detail=f"available: {', '.join(sorted(available)) or '(none)'}",
                    )
                )
    return n_lists, n_projected


def _check_follow_up_passes(
    spec: Dict[str, Any],
    *,
    key: str,
    rel: str,
    text: str,
    declared: Dict[str, Any],
    diags: List[Dict[str, Any]],
) -> None:
    """Error for every reason ``KnowledgePack.follow_up_passes`` would drop an entry.

    A dropped pass asks nothing and the run still reports success: the conditions reading the
    pass's source go ``unknown``, exactly as they do when the source returned nothing. Every
    drop reason is an error here, where the author is looking.

    Field paths are not checked here; they ride into ``_check_field_paths`` with every other
    block against the harvest source's inventory. A harvest's ``where`` clauses are the
    exception: their paths ride along too, but a skipped clause leaves no missing artifact,
    so the declaration itself must be checked (:func:`_check_harvest_where`).
    """
    entries = spec.get("follow_up_passes")
    if entries is None:
        return
    if not isinstance(entries, list):
        diags.append(
            _diag(
                "error",
                "follow-up-not-a-list",
                f"ruleset {key!r} declares `follow_up_passes` as "
                f"{type(entries).__name__}, not a list, so every follow-up pass is ignored",
                path=rel,
                line=_find_key_line(text, "follow_up_passes"),
            )
        )
        return
    line = _find_key_line(text, "follow_up_passes")
    seen: Dict[int, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            diags.append(
                _diag(
                    "error",
                    "follow-up-not-a-mapping",
                    f"ruleset {key!r} has a `follow_up_passes` item that is not a mapping "
                    "and is ignored whole",
                    path=rel,
                    line=line,
                )
            )
            continue
        try:
            number = int(entry.get("pass", 0))
        except (TypeError, ValueError):
            number = 0
        if number < 2:
            diags.append(
                _diag(
                    "error",
                    "follow-up-bad-pass-number",
                    f"ruleset {key!r} declares a follow-up pass numbered "
                    f"{entry.get('pass')!r}; passes are 2-based (pass 1 is the ordinary "
                    "retrieval), so this entry is dropped and its source is never queried",
                    path=rel,
                    line=line,
                    hint="number the first follow-up pass 2",
                )
            )
        else:
            seen[number] = seen.get(number, 0) + 1
        # `source` may be one name or a list of names sharing one pass. Each is checked
        # separately: a stale name beside working ones lets the pass run while one source's
        # conditions read `unknown` with every stage reporting success.
        raw_target = entry.get("source")
        if isinstance(raw_target, (list, tuple, set)):
            targets = [str(t).strip() for t in raw_target if str(t).strip()]
        else:
            targets = [t for t in [str(raw_target or "").strip()] if t]
        if not targets:
            diags.append(
                _diag(
                    "error",
                    "follow-up-no-source",
                    f"ruleset {key!r} declares a follow-up pass with no `source`, so there "
                    "is nothing for it to retrieve and the entry is dropped",
                    path=rel,
                    line=line,
                )
            )
        for target in targets:
            if declared and target not in declared:
                diags.append(
                    _diag(
                        "error",
                        "follow-up-unknown-source",
                        f"follow-up pass {number} targets {target!r}, which this ruleset's "
                        "`sources:` map does not declare — the pass cannot resolve a physical "
                        "source and its conditions read `unknown`",
                        path=rel,
                        line=_find_key_line(text, "sources"),
                        hint="use the LOGICAL source name as spelled in `sources:`",
                    )
                )
        if len(targets) != len(set(targets)):
            diags.append(
                _diag(
                    "warning",
                    "follow-up-duplicate-target",
                    f"follow-up pass {number} names the same target more than once; it is "
                    "queried once, so the repetition changes nothing and reads as two "
                    "questions",
                    path=rel,
                    line=line,
                )
            )
        # An unreadable window mode is not a runtime error: `_follow_up_window` resolves
        # anything unrecognised to `inherit`, so a typo silently runs the incident's own
        # window instead of the declared one.
        mode = str(entry.get("window", "") or "inherit").strip().lower()
        if mode not in ("inherit", "onwards") and not mode.startswith("lookback"):
            diags.append(
                _diag(
                    "error",
                    "follow-up-unknown-window",
                    f"follow-up pass {number} declares `window: {entry.get('window')!r}`, "
                    "which the engine does not recognise — it resolves to `inherit`, so the "
                    "pass silently runs over the incident's own window",
                    path=rel,
                    line=line,
                    hint="one of `inherit`, `onwards`, `lookback:<N>d`",
                )
            )
        elif mode.startswith("lookback") and not re.match(
            r"^lookback\s*[:=]?\s*\d+\s*d?$", mode
        ):
            diags.append(
                _diag(
                    "error",
                    "follow-up-lookback-no-depth",
                    f"follow-up pass {number} declares `window: {entry.get('window')!r}` with "
                    "no usable depth, so it falls back to `inherit` and the antecedent "
                    "question is asked over the episode's window only",
                    path=rel,
                    line=line,
                    hint="spell the depth in days, e.g. `lookback:365d`",
                )
            )
        harvest = entry.get("harvest")
        items = [h for h in (harvest or []) if isinstance(h, dict)]
        # A `together:` item declares `entity`/`fields` per component, so an item-level test
        # reads a correct co-occurrence harvest as empty. The consequence is the function's
        # own headline error: an author told the pass is dropped when it runs.
        usable = [h for h in items if _harvest_reads_something(h)]
        if not usable:
            diags.append(
                _diag(
                    "error",
                    "follow-up-nothing-harvested",
                    f"follow-up pass {number} harvests nothing usable (an entry needs both "
                    "`entity` and `fields`, per component where it declares `together`), so "
                    "the pass is dropped — a pass with no harvested values would be a "
                    "window-only scan answering no condition",
                    path=rel,
                    line=line,
                )
            )
        # Over every item, not just the usable ones: a `together` the engine cannot read is
        # why an item stops being usable, and the error above says the pass is dropped without
        # naming which declaration dropped it.
        for item in items:
            _check_harvest_together(item, number=number, rel=rel, text=text, diags=diags)
        for item in usable:
            _check_harvest_where(item, number=number, rel=rel, text=text, diags=diags)
            from_source = str(item.get("source", "") or "").strip()
            if from_source and declared and from_source not in declared:
                diags.append(
                    _diag(
                        "warning",
                        "follow-up-harvest-unknown-source",
                        f"follow-up pass {number} harvests "
                        f"{_harvest_label(item)!r} from {from_source!r}, which this "
                        "ruleset's `sources:` map does not declare — it is read as a "
                        "physical source name, and if that is wrong the pass harvests "
                        "nothing and skips",
                        path=rel,
                        line=line,
                        hint=(
                            "use the LOGICAL name, or confirm the physical source is "
                            "retrieved on this run"
                        ),
                    )
                )
        raw_purpose = entry.get("purpose")
        if isinstance(raw_purpose, dict):
            for named in raw_purpose:
                if str(named).strip() not in targets:
                    diags.append(
                        _diag(
                            "warning",
                            "follow-up-purpose-unknown-target",
                            f"follow-up pass {number} declares a `purpose` for "
                            f"{str(named)!r}, which is not one of its targets — that source "
                            "is not queried by this pass, so the text is never used",
                            path=rel,
                            line=_find_key_line(text, "purpose"),
                            hint="key `purpose` by the LOGICAL names listed under `source`",
                        )
                    )
            for target in targets:
                if not str(raw_purpose.get(target, "") or "").strip():
                    diags.append(
                        _diag(
                            "warning",
                            "follow-up-purpose-missing-target",
                            f"follow-up pass {number} declares `purpose` per source but "
                            f"none for {target!r}, so that query is described by the "
                            "source's own catalog text and not by the question this pass "
                            "asks",
                            path=rel,
                            line=_find_key_line(text, "purpose"),
                        )
                    )
        capture = str(entry.get("capture", "") or "")
        if capture:
            try:
                re.compile(capture)
            except re.error as exc:
                diags.append(
                    _diag(
                        "error",
                        "follow-up-bad-capture",
                        f"follow-up pass {number} declares a `capture` regex that does not "
                        "compile, so no harvested value survives it and the pass skips",
                        path=rel,
                        line=_find_key_line(text, "capture"),
                        detail=str(exc),
                    )
                )
    for number, count in sorted(seen.items()):
        if count > 1:
            diags.append(
                _diag(
                    "error",
                    "duplicate-follow-up-pass",
                    f"ruleset {key!r} declares pass {number} {count} times; the engine runs "
                    "the FIRST and the others never happen",
                    path=rel,
                    line=line,
                )
            )


def _together_components(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The mappings under ``together:``, or ``[]``; mirrors ``follow_up._tuple_components``."""
    raw = item.get("together")
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, dict)]


def _harvest_reads_something(item: Dict[str, Any]) -> bool:
    """Whether the engine harvests anything at all from one item, either way it can be spelled.

    ``follow_up`` reads a `together:` item through its components and an ordinary one through
    its own ``entity``/``fields``, and a single-component `together:` is read the same way as
    an ordinary harvest, so "usable" is one predicate over both spellings.
    """

    def _declares(node: Dict[str, Any]) -> bool:
        return bool(
            str(node.get("entity", "") or "").strip()
            and [f for f in (node.get("fields") or []) if str(f).strip()]
        )

    return _declares(item) or any(_declares(c) for c in _together_components(item))


#: Minimum corpus size for a ``base_rate`` to count. A floor on the corpus, not on the fire
#: count: rate is meaningless below it. Deliberately low; the alternative to a thin corpus is
#: ``kind: stub``, and a threshold nobody can reach trains authors to declare the stub forever.
_MIN_BASE_RATE_CORPUS = 10


def _check_entry_signals(
    spec: Dict[str, Any],
    *,
    key: str,
    rel: str,
    text: str,
    declared: Dict[str, Any],
    physical: Set[str],
    diags: List[Dict[str, Any]],
) -> None:
    """Validate inbound cross-procedure signals declared under ``entry_signals``.

    An unusable signal is silent: the link assessment reports the sibling as
    ``not_probed``, indistinguishable from a procedure that declared nothing.

    * ``opens_with.entity`` must equal the ruleset's ``subject_entity``; a leg is
      opened by a subject value, so a mismatched type can never be acted on.
    * ``base_rate`` is wanted (counted or ``kind: stub``); a signal without a credible
      rate works but a reader cannot discount it.
    * ``when`` with no ``where`` fires on any retrieved row (retrieval test, not
      evidence); warning because a conditional source is legitimate.
    """
    entries = spec.get("entry_signals")
    if entries is None:
        return
    line = _find_key_line(text, "entry_signals")
    if not isinstance(entries, list):
        diags.append(
            _diag(
                "error",
                "entry-signal-not-a-list",
                f"ruleset {key!r} declares `entry_signals` as {type(entries).__name__}, "
                "not a list, so every inbound signal is ignored and this procedure can "
                "never be recognised in another procedure's evidence",
                path=rel,
                line=line,
            )
        )
        return
    subject = str(spec.get("subject_entity", "") or "").strip()
    seen_ids: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            diags.append(
                _diag(
                    "error",
                    "entry-signal-not-a-mapping",
                    f"ruleset {key!r} has an `entry_signals` item that is not a mapping "
                    "and is ignored whole",
                    path=rel,
                    line=line,
                )
            )
            continue
        signal_id = str(entry.get("id", "") or "").strip()
        at = _find_id_line(text, signal_id) if signal_id else line
        label = signal_id or "(unnamed)"
        if not signal_id:
            diags.append(
                _diag(
                    "error",
                    "entry-signal-no-id",
                    f"ruleset {key!r} declares an entry signal with no `id`, so it is "
                    "DROPPED — there would be nothing for a report to cite and nothing an "
                    "operator could act on",
                    path=rel,
                    line=line,
                    hint="give it a stable id; a report quotes it verbatim",
                )
            )
        elif signal_id in seen_ids:
            diags.append(
                _diag(
                    "warning",
                    "entry-signal-duplicate-id",
                    f"ruleset {key!r} declares two entry signals with id {signal_id!r}, so a "
                    "cited link cannot be traced back to one declaration",
                    path=rel,
                    line=at,
                )
            )
        seen_ids.add(signal_id)

        # --- the pivot: ``R2`` made mechanical -----------------------------------
        opens = entry.get("opens_with")
        opens = opens if isinstance(opens, dict) else {}
        entity = str(opens.get("entity", "") or "").strip()
        if not entity:
            diags.append(
                _diag(
                    "error",
                    "entry-signal-no-subject",
                    f"entry signal {label!r} declares no `opens_with.entity`, so it is "
                    "DROPPED — a signal states what a sibling run must already HOLD to open "
                    "this procedure, and without that there is no leg to open",
                    path=rel,
                    line=at,
                    hint=(
                        f"`opens_with: {{entity: {subject or '<this ruleset s subject>'}}}`"
                    ),
                )
            )
        elif subject and entity != subject:
            diags.append(
                _diag(
                    "error",
                    "entry-signal-subject-mismatch",
                    f"entry signal {label!r} opens with {entity!r} while ruleset {key!r} "
                    f"adjudicates {subject!r} — this procedure cannot iterate {entity!r}, so "
                    "the signal promises a referral that can never be acted on",
                    path=rel,
                    line=at,
                    hint=(
                        f"a leg is opened by a SUBJECT value: use {subject!r}, or change "
                        "this ruleset's `subject_entity`"
                    ),
                )
            )

        # --- the rows ---------------------------------------------------------
        when = entry.get("when")
        when = when if isinstance(when, dict) else {}
        source = str(when.get("source", "") or "").strip()
        if not source:
            diags.append(
                _diag(
                    "error",
                    "entry-signal-no-source",
                    f"entry signal {label!r} declares no `when.source`, so it is DROPPED — "
                    "there are no rows for it to read",
                    path=rel,
                    line=at,
                )
            )
        elif source not in declared and source not in physical:
            # Not resolvable either way: the accessor passes an unknown name through as a
            # physical one, so the signal is asked of a key no run produces and never fires.
            diags.append(
                _diag(
                    "error",
                    "entry-signal-unknown-source",
                    f"entry signal {label!r} reads {source!r}, which is neither in this "
                    f"ruleset's `sources:` map nor a source the catalog declares — it can "
                    "never match a retrieved source, so the signal can never fire",
                    path=rel,
                    line=at,
                    hint=(
                        "use this ruleset's own LOGICAL name, or the catalog's physical "
                        "name; either way the signal is read opportunistically and is NOT "
                        "retrieved for its own sake"
                    ),
                )
            )
        try:
            min_rows = int(when.get("min_rows", 1))
        except (TypeError, ValueError):
            min_rows = 0
        if min_rows < 1:
            diags.append(
                _diag(
                    "warning",
                    "entry-signal-bad-min-rows",
                    f"entry signal {label!r} declares `min_rows: "
                    f"{when.get('min_rows')!r}`, which the engine floors at 1 — so the "
                    "signal fires on a single row, and one row of an ordinary source is not "
                    "a detector",
                    path=rel,
                    line=at,
                    hint="state how many rows it takes; the count is the declaration",
                )
            )

        # --- the causal axis --------------------------------------------------
        directions = link_directions()
        direction = str(entry.get("direction", "") or "").strip().lower()
        if directions and direction not in directions:
            diags.append(
                _diag(
                    "error",
                    "entry-signal-bad-direction",
                    f"entry signal {label!r} declares `direction: "
                    f"{entry.get('direction')!r}`, which the engine does not recognise — the "
                    "link is reported with no causal reading and a referral gets no window, "
                    "so an operator cannot tell 'this may have CAUSED the incident' from "
                    "'the incident may have caused this'",
                    path=rel,
                    line=at,
                    hint=f"one of {', '.join(directions)}",
                )
            )
        window = str(entry.get("window", "") or "inherit").strip().lower()
        if window not in ("inherit", "onwards") and not re.match(
            r"^lookback\s*[:=]?\s*\d+\s*d?$", window
        ):
            # Same vocabulary and failure direction as a follow-up pass window: an unreadable
            # mode degrades to the episode's own window, so a referral asks the wrong period
            # silently.
            diags.append(
                _diag(
                    "error",
                    "entry-signal-bad-window",
                    f"entry signal {label!r} declares `window: {entry.get('window')!r}`, "
                    "which the engine does not recognise — a referral built from it falls "
                    "back to the incident's own window, so an antecedent question is asked "
                    "over the episode only",
                    path=rel,
                    line=at,
                    hint="one of `inherit`, `onwards`, `lookback:<N>d`",
                )
            )

        # --- how much the firing is worth, and how often it fires -------------
        try:
            strength = float(entry.get("strength", 0.0) or 0.0)
        except (TypeError, ValueError):
            strength = -1.0
        if not 0.0 <= strength <= 1.0:
            diags.append(
                _diag(
                    "warning",
                    "entry-signal-bad-strength",
                    f"entry signal {label!r} declares `strength: "
                    f"{entry.get('strength')!r}`, which is not a number in 0..1 — it reads "
                    "as 0, so the signal never clears a configured strength floor and never "
                    "carries an advisory severity",
                    path=rel,
                    line=at,
                )
            )
        rate = entry.get("base_rate")
        rate = rate if isinstance(rate, dict) else {}
        measured, corpus = _base_rate_state(rate)
        if not measured:
            # One code and severity but three states (omitted, stub, thin corpus) because
            # the remedies differ; each state needs its own message so a reader can act.
            if not rate:
                state = "declares no `base_rate` at all"
                remedy = (
                    "declare `base_rate: {fires_on: N, of: M, measured: <date>}` once the "
                    "pair is counted, or `base_rate: {kind: stub}` to say out loud that it "
                    "is not"
                )
            elif str(rate.get("kind", "") or "").strip().lower() == "stub":
                state = "declares `base_rate: {kind: stub}` — measurement not yet taken"
                remedy = (
                    "replace the stub with `{fires_on: N, of: M, measured: <date>}` once "
                    "the corpus can carry the pair; until then the signal is reported as "
                    "declared-but-unmeasured and never auto-probed"
                )
            else:
                state = (
                    f"declares a `base_rate` of corpus {corpus}, below the "
                    f"{_MIN_BASE_RATE_CORPUS} runs a rate is read from"
                    if corpus
                    else "declares a `base_rate` with no countable corpus"
                )
                remedy = (
                    "count the pair over more runs, or declare `base_rate: {kind: stub}` "
                    "until the corpus is there — a rate over three runs is not a rate"
                )
            diags.append(
                _diag(
                    "warning",
                    "entry-signal-unmeasured",
                    f"entry signal {label!r} {state} — a signal that fires on most runs "
                    "is not a detector, and a reader given no rate cannot discount one "
                    "that fired",
                    path=rel,
                    line=at,
                    hint=remedy,
                )
            )
        # --- the selector -----------------------------------------------------
        # Without `where`, the signal fires on any retrieved row (retrieval test, not
        # evidence). Warning rather than error: a source conditional on the relevant shape
        # is legitimate, and the engine cannot distinguish that from a missing clause.
        clauses = when.get("where")
        clauses = clauses if isinstance(clauses, list) else []
        selective = [c for c in clauses if isinstance(c, dict) and c]
        if source and not selective:
            asks = bool(entry.get("auto_probe", False))
            diags.append(
                _diag(
                    "warning",
                    "entry-signal-broad-selector",
                    f"entry signal {label!r} reads {source!r} with no `when.where` clause, so "
                    f"it fires on any {min_rows if min_rows > 1 else 1} row(s) of that source — "
                    "the test is that the source was retrieved at all"
                    + (", and `auto_probe: true` spends a query on it" if asks else ""),
                    path=rel,
                    line=at,
                    hint=(
                        "name the rows that mean this procedure's fraud — a `where` clause, or "
                        "a `min_rows` a normal run does not reach; a selector this broad makes "
                        "the signal a fact about retrieval rather than about the evidence"
                    ),
                )
            )


def _base_rate_state(rate: Dict[str, Any]) -> Tuple[bool, int]:
    """``(credible, corpus size)`` for one ``base_rate`` block, delegated to the engine.

    Delegated rather than restated: ``src/link_escalation.py`` asks this same question of the
    same block, and both answers must be the same answer. A separate implementation would be
    a threshold with two homes.
    """
    return link_escalation.base_rate_measured(rate)


def _check_link_escalation(
    spec: Dict[str, Any],
    *,
    key: str,
    rel: str,
    text: str,
    ruleset_keys: Set[str],
    diags: List[Dict[str, Any]],
) -> None:
    """Validate ``link_escalation`` on one ruleset.

    Three ways to get it wrong, all silent in the same direction (the pair silently falls back
    to ``planned``):

    * An unspellable mode is dropped. Error: the vocabulary is closed and decidable.
    * A ``from:`` key naming no known ruleset can never match. Error: the key is another
      procedure's name, the one string an author of this file cannot verify by reading it.
    * An escalating mode on a ruleset with no ``gate: scope`` condition is unreachable. Rung-1
      pass on the sibling run's rows licenses an escalation, and a ruleset with no gate has no
      rung 1. Error: decidable from this file alone.
    """
    declared = spec.get("link_escalation")
    if declared is None:
        return
    line = _find_key_line(text, "link_escalation")
    if not isinstance(declared, dict):
        diags.append(
            _diag(
                "error",
                "link-escalation-not-a-mapping",
                f"ruleset {key!r} declares `link_escalation` as "
                f"{type(declared).__name__}, not a mapping, so it is ignored and every link "
                "to this procedure stays at "
                f"{link_escalation.DEFAULT_LINK_MODE!r}",
                path=rel,
                line=line,
                hint="`link_escalation: {mode: <mode>, from: {<use case>: <mode>}}`",
            )
        )
        return

    # Every mode this block declares, each with the label the author will recognise it by.
    asked: List[Tuple[str, Any]] = [("mode", declared.get("mode"))]
    raw_from = declared.get("from")
    if raw_from is not None and not isinstance(raw_from, dict):
        diags.append(
            _diag(
                "error",
                "link-escalation-bad-from",
                f"ruleset {key!r} declares `link_escalation.from` as "
                f"{type(raw_from).__name__}, not a mapping of source procedure to mode, so "
                "every per-source override is ignored",
                path=rel,
                line=line,
                hint="`from: {<source use case>: planned|semi_auto|auto}`",
            )
        )
        raw_from = {}
    for source, mode in (raw_from or {}).items():
        asked.append((f"from.{source}", mode))
        if str(source) not in ruleset_keys:
            diags.append(
                _diag(
                    "error",
                    "link-escalation-unknown-source",
                    f"ruleset {key!r} declares `link_escalation.from.{source}` but no "
                    f"ruleset named {str(source)!r} exists in this pack, so the override can "
                    "never match and the pair silently takes the general mode",
                    path=rel,
                    line=line,
                    hint=(
                        "name a ruleset key — "
                        + ", ".join(sorted(ruleset_keys)[:8])
                        + (" …" if len(ruleset_keys) > 8 else "")
                    ),
                )
            )

    escalating = False
    for label, mode in asked:
        if mode is None or not str(mode).strip():
            continue
        resolved = link_escalation.normalise_mode(mode)
        if not resolved:
            diags.append(
                _diag(
                    "error",
                    "link-escalation-bad-mode",
                    f"ruleset {key!r} declares `link_escalation.{label}: {mode!r}`, which is "
                    "not a mode this engine recognises, so it is dropped and the pair reads "
                    "as one that declared nothing",
                    path=rel,
                    line=line,
                    hint="one of: " + ", ".join(link_escalation.LINK_MODES),
                )
            )
            continue
        escalating = escalating or resolved in link_escalation.ESCALATING_MODES

    if not escalating:
        return
    # Check the licence exactly as the runtime clamp does: rung 1 is this ruleset's own
    # scope gate re-evaluated against a sibling run's rows. No `gate: scope` condition means
    # no rung 1 to pass, so the escalating mode is unreachable on every incident.
    gated = any(
        isinstance(c, dict) and str(c.get("gate", "") or "").strip().lower() == "scope"
        for c in (spec.get("conditions") or [])
        if isinstance(spec.get("conditions"), list)
    )
    if gated:
        return
    diags.append(
        _diag(
            "error",
            "link-escalation-no-scope-gate",
            f"ruleset {key!r} declares an escalating `link_escalation` mode but no condition "
            "carrying `gate: scope`, so it has no applicability test for a sibling run to "
            "evaluate — the one thing that licenses an automatic escalation can never hold, and "
            f"every link to this procedure is held at {link_escalation.MANUAL_LINK_MODE!r} on "
            "every incident",
            path=rel,
            line=line,
            hint=(
                "mark the condition(s) that decide whether this procedure applies at all with "
                "`gate: scope`, or declare the mode as "
                f"{link_escalation.MANUAL_LINK_MODE!r} — an escalating mode is unlocked by this "
                "ruleset's own gate holding on the run's evidence, not by declaring it"
            ),
        )
    )


def _playbook_link_graph(root: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """``({playbook id: use case}, {playbook id: [related playbook ids]})`` from frontmatter.

    Read here rather than through the loader for the reason the whole module exists: the loader
    turns a broken frontmatter block into ``{}``, so a pack whose graph was destroyed by a
    parse error would validate as a pack that declared no graph.
    """
    ids: Dict[str, str] = {}
    related: Dict[str, List[str]] = {}
    for path in sorted(root.rglob("playbooks/*.md")):
        # The use case is stamped from the directory, not frontmatter (`_read_markdown_docs`
        # takes it as an argument). Reading a frontmatter `use_case:` key would map every
        # playbook to no procedure and silence every check below.
        parts = path.relative_to(root).parts
        from_dir = parts[1] if len(parts) > 3 and parts[0] == "use_cases" else ""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        front: Any = {}
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                try:
                    front = yaml.safe_load(text[3:end])
                except yaml.YAMLError:
                    front = {}  # already reported by `_check_frontmatter`
        front = front if isinstance(front, dict) else {}
        # A playbook with no frontmatter is still a playbook and is still a legal target: the
        # loader falls back to the file stem for its id, so registering it here is what keeps a
        # relationship pointing at it from being reported as naming nothing.
        pid = str(front.get("playbook_id", "") or path.stem).strip()
        ids[pid] = from_dir or str(front.get("use_case", "") or "").strip()
        rel_list = front.get("related_playbooks")
        if isinstance(rel_list, list):
            related[pid] = [str(v).strip() for v in rel_list if str(v).strip()]
    return ids, related


def _co_binding_map(
    catalog: Any, names: Optional[Set[str]] = None
) -> Dict[str, Set[str]]:
    """``{entity type: types co-bound with it on some source}``: the pivot-conversion graph.

    The same computation as ``KnowledgePack.entity_binding_map``, over the raw catalog.
    Recomputed here rather than imported because this module reads the pack's bytes and never
    the loaded object.

    ``names`` narrows it to those sources. The two scopes answer different questions: unscoped
    is "could a query anywhere in this pack bind both"; scoped to one procedure's declared
    sources is "does this procedure's own evidence hand the value back". The second is a
    strict subset.
    """
    out: Dict[str, Set[str]] = {}
    for src in (catalog or {}).get("sources") or []:
        if not isinstance(src, dict):
            continue
        if names is not None and str(src.get("name", "")) not in names:
            continue
        bound = {str(t) for t in (src.get("entity_bindings") or {})}
        for etype in bound:
            out.setdefault(etype, set()).update(bound - {etype})
    return out


def _check_link_graph(
    root: Path,
    catalog: Any,
    rulesets: Dict[str, Dict[str, Any]],
    diags: List[Dict[str, Any]],
    counts: Dict[str, int],
) -> None:
    """Reconcile the prose link graph against the operative one, both directions as info.

    Two graphs describe which procedures relate:

    * ``related_playbooks:`` in a playbook's frontmatter: written by a human.
    * ``entry_signals`` + ``entity_bindings``: what the engine can act on (a leg exists
      only where some source binds both procedures' subject types on one row).

    Where they disagree, both directions are reported as info: the disagreement is named
    without guessing which side moved (a new signal may discover an undeclared edge;
    a prose edge may be a relationship the data cannot execute).
    """
    ids, related = _playbook_link_graph(root)
    if not related and not any(r.get("entry_signals") for r in rulesets.values()):
        return
    reach = _co_binding_map(catalog)
    subjects = {
        key: str(r.get("subject_entity", "") or "").strip()
        for key, r in rulesets.items()
    }

    # use case -> the use cases its playbooks claim a relationship with, both ways
    claimed: Dict[str, Set[str]] = {}
    legs = 0
    for pid, targets in sorted(related.items()):
        source_uc = ids.get(pid, "")
        for target in targets:
            legs += 1
            target_uc = ids.get(target, "")
            if source_uc:
                claimed.setdefault(source_uc, set()).add(target_uc or target)
            if target_uc:
                claimed.setdefault(target_uc, set()).add(source_uc or pid)
            if target not in ids:
                # Its own code, not `unreachable`: a leg nothing can bind and a leg naming
                # nothing are different defects with different fixes. A collective phrase in
                # a list of ids reaches the engine as an unresolvable id.
                diags.append(
                    _diag(
                        "info",
                        "related-playbook-unknown",
                        f"playbook {pid!r} declares a relationship with {target!r}, which "
                        "names no playbook in this pack — a reader can act on it, the link "
                        "assessment cannot",
                        path=str(root.name),
                        hint=(
                            "list the playbook ids; a collective phrase reaches the engine as "
                            "one unresolvable id"
                        ),
                    )
                )
                continue
            sa, sb = subjects.get(source_uc, ""), subjects.get(target_uc, "")
            if not target_uc or target_uc not in rulesets:
                continue  # narrative-only procedure: nothing adjudicates it, by design
            if not sa or not sb or sa == sb:
                continue
            if sb in reach.get(sa, set()):
                # Two cases: co-bound on a source this procedure already retrieves (free
                # at rung 0) versus co-bound only on a source it does not (costs a query).
                # Reporting them the same way makes an unopened leg look answered.
                mine = (rulesets.get(source_uc) or {}).get("sources") or {}
                own_sources = {str(v) for v in mine.values()}
                if sb not in _co_binding_map(catalog, own_sources).get(sa, set()):
                    diags.append(
                        _diag(
                            "info",
                            "related-playbook-needs-probe",
                            f"playbook {pid!r} declares a relationship with {target!r}, and "
                            f"{sa!r} does co-bind with {sb!r} — but on NO source "
                            f"{source_uc!r} retrieves, so this leg cannot be opened from its "
                            "own evidence and needs a query of a source it does not ask",
                            path=str(root.name),
                            hint=(
                                "the leg is real and is not free: it is reported as a "
                                "candidate needing one probe, never as a checked-and-clear "
                                "link"
                            ),
                        )
                    )
                continue
            diags.append(
                _diag(
                    "info",
                    "related-playbook-unreachable",
                    f"playbook {pid!r} declares a relationship with {target!r}, but no source "
                    f"binds {sa!r} and {sb!r} on one row — so a run of {source_uc!r} can never "
                    f"produce the {sb!r} value {target_uc!r} needs, and the leg is reported as "
                    "a binding gap rather than as a candidate",
                    path=str(root.name),
                    hint=(
                        f"bind {sb!r} on a source that already binds {sa!r}, or accept the gap "
                        "— a stated limitation is the deliverable here, not a defect"
                    ),
                )
            )
    counts["related_playbook_legs"] = legs

    # Which procedures retrieve each physical source: that determines who an inbound signal
    # can be read on, i.e. the operative edges the prose graph is compared against.
    declarers: Dict[str, Set[str]] = {}
    for key, spec in rulesets.items():
        for phys in (spec.get("sources") or {}).values():
            if isinstance(phys, str):
                declarers.setdefault(phys, set()).add(key)
    for key, spec in sorted(rulesets.items()):
        own = spec.get("sources")
        own = own if isinstance(own, dict) else {}
        for entry in spec.get("entry_signals") or []:
            if not isinstance(entry, dict):
                continue
            when = entry.get("when") if isinstance(entry.get("when"), dict) else {}
            logical = str(when.get("source", "") or "").strip()
            phys = str(own.get(logical, logical) or "")
            for other in sorted(declarers.get(phys, set()) - {key}):
                if key in claimed.get(other, set()) or other in claimed.get(key, set()):
                    continue
                diags.append(
                    _diag(
                        "info",
                        "entry-signal-not-related",
                        f"ruleset {key!r} can be recognised in {other!r}'s evidence (both read "
                        f"{phys!r}), and neither playbook declares the other under "
                        "`related_playbooks` — the operative graph has an edge the prose graph "
                        "does not",
                        path=str(root.name),
                        hint=(
                            "add the edge to the playbook if it is real, or narrow the "
                            "signal's source if it is not — a signal read on a procedure "
                            "nobody related it to is where a false referral comes from"
                        ),
                    )
                )


def _harvest_label(item: Dict[str, Any]) -> str:
    """The name to use for one harvest in a diagnostic, regardless of which spelling it uses.

    A ``together:`` item has no entity of its own, and a message naming ``None`` sends an
    author looking for a key that is not the one at fault.
    """
    own = str(item.get("entity", "") or "").strip()
    if own:
        return own
    named = [
        str(c.get("entity", "") or "").strip()
        for c in _together_components(item)
        if str(c.get("entity", "") or "").strip()
    ]
    return "+".join(named) if named else str(item.get("entity"))


def _check_harvest_together(
    item: Dict[str, Any],
    *,
    number: int,
    rel: str,
    text: str,
    diags: List[Dict[str, Any]],
) -> None:
    """Validate ``together:`` in a harvest item.

    A discarded ``together:`` degrades to per-type lists: the query runs, more rows come back
    than the evidence justifies, the row cap truncates on other parties' rows, and the
    procedure adjudicates the mixture as the subject's conduct. These are errors rather than
    warnings because the engine only logs the drop at runtime and authoring time is the only
    place the difference is visible.
    """
    raw = item.get("together")
    if raw is None:
        return
    label = _harvest_label(item)
    line = _find_key_line(text, "together")
    if not isinstance(raw, list) or not raw:
        diags.append(
            _diag(
                "error",
                "harvest-together-not-a-list",
                f"follow-up pass {number}'s {label!r} harvest declares `together` as "
                f"{type(raw).__name__}, not a non-empty list, so the co-occurrence is "
                "discarded whole — the item harvests only what its own `entity`/`fields` "
                "declare, and the follow-up query asks the CROSS PRODUCT of the lists it does "
                "carry",
                path=rel,
                line=line,
                hint="a list of `{entity, fields}` mappings, one per component",
            )
        )
        return
    components = _together_components(item)
    if len(components) != len(raw):
        diags.append(
            _diag(
                "error",
                "harvest-together-not-a-mapping",
                f"follow-up pass {number}'s {label!r} harvest has a `together` component that "
                "is not a mapping and is skipped, so that component's values are never "
                "harvested and the combination is enforced without it",
                path=rel,
                line=line,
            )
        )
    complete: List[Dict[str, Any]] = []
    for comp in components:
        entity = str(comp.get("entity", "") or "").strip()
        fields = [f for f in (comp.get("fields") or []) if str(f).strip()]
        if not entity or not fields:
            missing = " and ".join(
                part
                for part in ("`entity`" if not entity else "", "`fields`" if not fields else "")
                if part
            )
            diags.append(
                _diag(
                    "error",
                    "harvest-together-incomplete-component",
                    f"follow-up pass {number}'s {label!r} harvest declares a `together` "
                    f"component with no {missing}, so it is dropped: its values are never "
                    "harvested, and the surviving components are asked without it",
                    path=rel,
                    line=line,
                    hint="every component needs both `entity` and `fields`",
                )
            )
            continue
        if str(comp.get("source", "") or "").strip():
            diags.append(
                _diag(
                    "error",
                    "harvest-together-component-source",
                    f"follow-up pass {number}'s {entity!r} `together` component declares its "
                    "own `source`, which the engine IGNORES — values from two sources did not "
                    "occur together in any row, so honouring it would invent exactly the "
                    "combinations this mechanism exists to prevent",
                    path=rel,
                    line=_find_key_line(text, "source"),
                    hint=(
                        "co-occurrence is defined within ONE source's rows; put `source` on "
                        "the harvest item"
                    ),
                )
            )
        complete.append(comp)
    entities = [str(c.get("entity", "") or "").strip() for c in complete]
    for entity in sorted({e for e in entities if entities.count(e) > 1}):
        diags.append(
            _diag(
                "error",
                "harvest-together-duplicate-entity",
                f"follow-up pass {number}'s {label!r} harvest declares {entity!r} as two "
                "`together` components; both values route to the one column the target binds "
                "for that type, so the combination collapses to a single column and cannot be "
                "enforced — the values are asked as a flat list",
                path=rel,
                line=line,
                hint=(
                    "one component per entity TYPE; a second field of the same type belongs "
                    "in that component's `fields`"
                ),
            )
        )
    if len(complete) == 1 and len(components) == 1:
        diags.append(
            _diag(
                "warning",
                "harvest-together-single-component",
                f"follow-up pass {number}'s {label!r} harvest declares one `together` "
                "component, which is an ordinary harvest spelled as a combination: one column "
                "is what the per-type list already constrains, so there is nothing for the "
                "guard to narrow",
                path=rel,
                line=line,
                hint="declare the second component, or drop `together`",
            )
        )


def _check_harvest_where(
    item: Dict[str, Any],
    *,
    number: int,
    rel: str,
    text: str,
    diags: List[Dict[str, Any]],
) -> None:
    """Error when a harvest ``where`` clause will be skipped by the engine.

    ``correlation.apply_where`` skips a clause naming no field or no values, because an
    incomplete clause emptying the row set looks downstream exactly like a source that
    returned nothing. The cost: the harvest reads every row, carries ordinary values beside
    the ones the incident is about, and the follow-up question is asked about both. An error,
    not a warning: unlike a dropped pass, there is no missing artifact.

    ``match`` is also checked: anything other than ``"exact"`` is treated as substring, so a
    typo silently widens short vocabularies. Field paths are not checked here; they ride into
    ``_check_field_paths`` against the harvest item's own ``source``.
    """
    where = item.get("where")
    if where is None:
        return
    entity = _harvest_label(item)
    line = _find_key_line(text, "where")
    if not isinstance(where, list):
        diags.append(
            _diag(
                "error",
                "harvest-where-not-a-list",
                f"follow-up pass {number} declares the {entity!r} harvest's `where` as "
                f"{type(where).__name__}, not a list, so it is ignored and the harvest reads "
                "EVERY row",
                path=rel,
                line=line,
            )
        )
        return
    for clause in where:
        if not isinstance(clause, dict):
            diags.append(
                _diag(
                    "error",
                    "harvest-where-not-a-mapping",
                    f"follow-up pass {number} has a {entity!r} harvest `where` item that is "
                    "not a mapping and is skipped, so those rows are not excluded",
                    path=rel,
                    line=line,
                )
            )
            continue
        field = str(clause.get("field") or "").strip()
        values = [v for v in (clause.get("any_of") or []) if str(v).strip()]
        if not field or not values:
            missing = " and ".join(
                part
                for part in (
                    "`field`" if not field else "",
                    "`any_of`" if not values else "",
                )
                if part
            )
            diags.append(
                _diag(
                    "error",
                    "harvest-where-incomplete",
                    f"follow-up pass {number}'s {entity!r} harvest declares a `where` clause "
                    f"with no {missing}, so it is SKIPPED and the harvest reads every row — "
                    "carrying the values this pass was scoped to exclude",
                    path=rel,
                    line=line,
                    hint="a clause needs both `field` and a non-empty `any_of`",
                )
            )
            continue
        match = str(clause.get("match", "exact"))
        if match.lower() not in ("exact", "substring"):
            diags.append(
                _diag(
                    "error",
                    "harvest-where-bad-match",
                    f"follow-up pass {number}'s {entity!r} harvest declares "
                    f"match: {match!r}; only `exact` is recognised and anything else is read "
                    "as `substring`, which on a short vocabulary matches far more rows than "
                    "the author asked for",
                    path=rel,
                    line=_find_key_line(text, "match"),
                    hint="`exact` or `substring`",
                )
            )


def _check_indicator_threshold(
    spec: Dict[str, Any],
    *,
    key: str,
    rel: str,
    text: str,
    diags: List[Dict[str, Any]],
) -> None:
    """Error when a ruleset's corroborating indicators have no ``indicator_threshold``.

    Without a threshold the engine refuses to weigh indicators and notes they fired
    unweighed. A threshold of zero is its own case: ``len([]) >= 0`` is true, so it
    reaches the fraud label with no indicator firing on every subject; reported
    separately from an absent key. Scoped to rulesets declaring non-decisive indicators;
    a decisive indicator's fail is the finding and needs no threshold.
    """
    voting = [
        c
        for c in (spec.get("conditions") or [])
        if isinstance(c, dict)
        and str(c.get("polarity", "") or "") == "fraud_indicator"
        and not c.get("decisive")
    ]
    if not voting:
        return
    raw = spec.get("indicator_threshold", None)
    declared = (
        int(raw)
        if isinstance(raw, int) and not isinstance(raw, bool)
        else (
            int(raw)
            if isinstance(raw, str) and raw.strip().isdigit()
            else None
        )
    )
    line = _find_key_line(text, "indicator_threshold") or _find_key_line(text, key)
    if declared is None:
        diags.append(
            _diag(
                "error",
                "indicator-threshold-undeclared",
                f"ruleset {key!r} declares {len(voting)} corroborating fraud indicator(s) "
                "and no `indicator_threshold`, so the engine has no rule for weighing them "
                "and will not weigh them at all",
                path=rel,
                line=line,
                detail=(
                    f"indicator_threshold: {raw!r}"
                    if raw is not None
                    else "no `indicator_threshold` key on the ruleset"
                ),
                hint="declare how many indicators must corroborate, e.g. "
                "`indicator_threshold: 2`",
            )
        )
        return
    if declared < 1:
        diags.append(
            _diag(
                "error",
                "indicator-threshold-vacuous",
                f"ruleset {key!r} sets `indicator_threshold: {declared}`, which is reached "
                "by zero indicators — so every subject it adjudicates gets the fraud label, "
                "including one whose every check passed",
                path=rel,
                line=line,
                hint="a corroboration threshold is at least 1",
            )
        )
        return
    if declared > len(voting):
        diags.append(
            _diag(
                "warning",
                "indicator-threshold-unreachable",
                f"ruleset {key!r} needs {declared} corroborating indicators and declares "
                f"only {len(voting)}, so the corroborated path can never be taken",
                path=rel,
                line=line,
                hint="lower the threshold, or drop it if this procedure is meant to reach "
                "its verdict some other way",
            )
        )


#: Condition kinds where the finding is a comparison against a ruleset-declared bound.
#: A count takes an integer; an interval takes the engine's ``<N>h``/``<N>d``/``<N>m`` form.
#: Keyed by kind so a new counting kind must choose explicitly whether its bound is mandatory.
_BOUNDED_KINDS = {
    "distinct_count": "integer",
    "velocity_count": "integer",
    "time_gap": "window",
    "numeric_compare": "numeric",
    "value_equivalence": "numeric",
}

#: The key each bounded kind declares its bound under. ``numeric_compare`` says ``bound``
#: rather than ``max`` because ``max`` states the wrong thing under ``operator: ">="``.
_BOUND_KEYS = {"numeric_compare": "bound", "value_equivalence": "bound"}

_WINDOW_RE = re.compile(r"^\s*\d+\s*[hdm]\s*$", re.IGNORECASE)


def _check_condition_bound(
    cond: Dict[str, Any],
    *,
    cid: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Error when a counting or interval condition has no readable ``max`` bound.

    The bound is the entire finding for these kinds: the rows supply a number and everything
    the report states comes from the declaration. An absent or unreadable bound makes the
    condition return ``unknown``, and an inert decisive check produces a verdict of
    ``INSUFFICIENT DATA``. Error: there is no reading under which a pack intended a comparison
    with nothing on one side.
    """
    kind = str(cond.get("kind", "") or "")
    grammar = _BOUNDED_KINDS.get(kind)
    if not grammar:
        return
    bound_key = _BOUND_KEYS.get(kind, "max")
    raw = cond.get(bound_key, None)
    if grammar == "integer":
        ok = not isinstance(raw, bool) and isinstance(raw, int)
        if not ok and isinstance(raw, str):
            ok = raw.strip().lstrip("-").isdigit()
        expected = f"a whole number, e.g. `{bound_key}: 1`"
    elif grammar == "numeric":
        # A fraction is the point of this grammar: a ratio bound of 0.25 read as an integer
        # becomes 0, a bound every value satisfies.
        ok = not isinstance(raw, bool) and isinstance(raw, (int, float))
        if not ok and isinstance(raw, str):
            try:
                float(raw.strip())
                ok = True
            except ValueError:
                ok = False
        expected = f"a number, whole or fractional, e.g. `{bound_key}: 3` / `{bound_key}: 0.25`"
    else:
        ok = isinstance(raw, str) and bool(_WINDOW_RE.match(raw))
        expected = (
            f"a window in the engine's grammar, e.g. `{bound_key}: 90m` / `4h` / `2d`"
        )
    if ok:
        return
    diags.append(
        _diag(
            "error",
            "condition-bound-undeclared",
            f"{cid or '(unnamed)'}: a {kind!r} check is a comparison against a "
            "declared bound, and this one declares none the engine can read — so it can "
            "only ever report `unknown`",
            path=rel,
            line=line,
            detail=(
                f"{bound_key}: {raw!r}"
                if raw is not None
                else f"no `{bound_key}` key on the condition"
            ),
            hint=f"declare `{bound_key}` as {expected}",
        )
    )


#: Keys that decide what a condition's finding is WORTH. The parent of a composite owns every
#: one of them: `mk()` reads them off the outer dict only, so a child declaring one is inert.
_WEIGHTING_KEYS = (
    "decisive",
    "decisive_on",
    "polarity",
    "exclusion_kind",
    "report_group",
    "order",
    "gate",
    "subject_scope",
    "row_match",
)


def _check_numeric_compare(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Check a ``numeric_compare``'s aggregate, operator, and the one key it does not read.

    All three failures are silent at run time: an unreadable aggregate or operator makes the
    condition ``unknown`` (a decisive one then reads as INSUFFICIENT DATA), and a declared
    ``exclude_subject`` reads as honoured while the subject's own rows stay in the aggregate.
    """
    if kind != "numeric_compare":
        return
    aggregates = _engine_literals("_AGGREGATES")
    agg = str(cond.get("aggregate", "") or "").strip().lower()
    if aggregates and agg not in aggregates:
        diags.append(
            _diag(
                "error",
                "numeric-compare-aggregate",
                f"{cid or '(unnamed)'}: `aggregate` is {agg or '(missing)'!r}, which the "
                "evaluator does not compute, so this comparison can only report `unknown`",
                path=rel,
                line=line,
                detail=f"available: {', '.join(sorted(aggregates))}",
            )
        )
    elif agg in _engine_literals("_MODAL_AGGREGATES") and not any(
        str(c.get("field", "") or "").strip()
        for c in [cond]
        + [f for f in (cond.get("fallbacks") or []) if isinstance(f, dict)]
    ):
        diags.append(
            _diag(
                "error",
                "numeric-compare-modal-field",
                f"{cid or '(unnamed)'}: `aggregate: {agg}` asks which value is most frequent "
                "and no `field` names the values, so the evaluator has nothing to rank and "
                "reports `unknown`",
                path=rel,
                line=line,
                hint=(
                    "declare `field:` naming the column whose concentration is in question; "
                    "`aggregate: count` is the one that reads the rows themselves"
                ),
            )
        )
    operators = compare_operators()
    op = str(cond.get("operator", "") or "").strip()
    if operators and op not in operators:
        diags.append(
            _diag(
                "error",
                "numeric-compare-operator",
                f"{cid or '(unnamed)'}: `operator` is {op or '(missing)'!r}, which the "
                "evaluator does not implement, so this comparison can only report `unknown`",
                path=rel,
                line=line,
                detail=f"available: {', '.join(operators)}",
            )
        )
    if cond.get("exclude_subject"):
        diags.append(
            _diag(
                "error",
                "numeric-compare-exclude-subject",
                f"{cid or '(unnamed)'}: `exclude_subject` is not read by "
                "`numeric_compare`, so the subject's own rows stay in the aggregate while "
                "the declaration says they were removed",
                path=rel,
                line=line,
                hint=(
                    "use `distinct_count`, which implements it, or exclude the subject with "
                    "a `where` clause on the values that identify it"
                ),
            )
        )
    if cond.get("group_by") and op == "==":
        diags.append(
            _diag(
                "warning",
                "numeric-compare-group-equality",
                f"{cid or '(unnamed)'}: `group_by` under `operator: \"==\"` names no "
                "deciding group — the evaluator compares the LARGEST group, which answers "
                "an upper bound and not an equality, so this check reports `unknown`",
                path=rel,
                line=line,
                hint="compare with `>=` / `<=`, or drop `group_by` to aggregate over all rows",
            )
        )


def _check_value_equivalence(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Check the two keys a ``value_equivalence`` cannot run without besides its bound.

    The relation is the whole finding here: with no ``form`` there is nothing to be equivalent
    UNDER, and the engine refuses rather than falling back to equality — so the condition can
    only report ``unknown``, and a decisive one then reads as INSUFFICIENT DATA.
    """
    if kind != "value_equivalence":
        return
    if not str(cond.get("form", "") or "").strip():
        diags.append(
            _diag(
                "error",
                "value-equivalence-form-undeclared",
                f"{cid or '(unnamed)'}: names no `form`, and the engine owns no relation of "
                "its own — what counts as the same thing is the pack's declaration, so this "
                "check can only report `unknown`",
                path=rel,
                line=line,
                hint=(
                    "declare `form:` naming an entry in shared/equivalence_forms.yaml; a "
                    "silent fallback to equality is the one wrong answer that still looks "
                    "like a working check"
                ),
            )
        )
    operators = compare_operators()
    op = str(cond.get("operator", "") or "").strip()
    if operators and op not in operators:
        diags.append(
            _diag(
                "error",
                "value-equivalence-operator",
                f"{cid or '(unnamed)'}: `operator` is {op or '(missing)'!r}, which the "
                "evaluator does not implement, so this comparison can only report `unknown`",
                path=rel,
                line=line,
                detail=f"available: {', '.join(operators)}",
            )
        )
    anchor = cond.get("anchor")
    if anchor is not None and (
        not isinstance(anchor, dict)
        or not str(anchor.get("source", "") or "")
        or not str(anchor.get("field", "") or "")
    ):
        diags.append(
            _diag(
                "error",
                "value-equivalence-anchor-shape",
                f"{cid or '(unnamed)'}: `anchor` needs both a `source` and a `field` naming "
                "the side the other values are compared against, and this declares "
                f"{anchor!r} — with no readable anchor there is nothing to be equivalent TO",
                path=rel,
                line=line,
            )
        )


def _check_condition_form(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    forms: Dict[str, Dict[str, Any]],
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Check the two keys a condition names a text-equivalence form under.

    ``form`` is the relation a ``value_equivalence`` adjudicates; ``normalize`` replaces the
    dedup key a counting kind reads. Both fail silently in the direction that matters: naming a
    form nobody declares, or naming one on a kind that does not read the key, leaves the
    condition comparing on the engine's incumbent reading while the pack believes otherwise.
    """
    for key in ("form", "normalize"):
        name = str(cond.get(key, "") or "").strip()
        if not name:
            continue
        # The strict derivation, not `kinds_reading`: the bare word "form" appears in half the
        # evaluator's prose and inside `norm_form`, which would attribute the key to kinds that
        # never read it and turn this warning into silence where it matters.
        readers = kinds_getting(key) | (
            form_normalize_kinds() if key == "normalize" else set()
        )
        if readers and kind not in readers:
            diags.append(
                _diag(
                    "warning",
                    "equivalence-form-noop-kind",
                    f"{cid or '(unnamed)'}: `{key}` is a no-op on kind {kind!r}, so this "
                    "check compares on the engine's own reading and not the declared form",
                    path=rel,
                    line=line,
                    detail="read only by: " + ", ".join(sorted(readers)),
                )
            )
            continue
        # `normalize` is one key over two vocabularies: the counting kinds resolve it against
        # the forms file, the older seams read a fixed mode name. A form name on a mode seam is
        # ignored, so the check runs on the engine's reading under the pack's own form name.
        if key == "normalize" and forms and kind not in form_normalize_kinds():
            if name in forms:
                diags.append(
                    _diag(
                        "error",
                        "equivalence-form-on-mode-seam",
                        f"{cid or '(unnamed)'}: `normalize: {name}` names a declared "
                        f"equivalence form, and {kind!r} reads `normalize` as one of its own "
                        "fixed mode names instead — the form is ignored and the comparison "
                        "runs on the engine's reading under the form's name",
                        path=rel,
                        line=line,
                        hint=(
                            "name the mode this kind reads, or ask the question through a kind "
                            "that resolves forms: "
                            + (", ".join(sorted(form_normalize_kinds())) or "(none)")
                        ),
                    )
                )
            continue
        if forms and name not in forms:
            diags.append(
                _diag(
                    "error",
                    "unknown-equivalence-form",
                    f"{cid or '(unnamed)'}: names the form {name!r}, and "
                    "shared/equivalence_forms.yaml declares no such form — the engine refuses "
                    "it rather than falling back, so this check can only report `unknown`",
                    path=rel,
                    line=line,
                    detail=f"declared: {', '.join(sorted(forms)) or '(none)'}",
                )
            )
            continue
        form = forms.get(name) or {}
        pairwise = bool(form.get("compare"))
        if key == "normalize" and pairwise:
            # A projection is transitive so "how many distinct" has one answer; a `compare`
            # relation is not, and the same question becomes "how many classes under which
            # linkage". Applying the projection half alone answers it under the form's name.
            diags.append(
                _diag(
                    "error",
                    "equivalence-form-not-transitive",
                    f"{cid or '(unnamed)'}: `normalize: {name}` names a form declaring "
                    "`compare`, which is not transitive and so has no distinct count of its "
                    "own — the engine refuses it",
                    path=rel,
                    line=line,
                    hint=(
                        "ask it through a `value_equivalence` condition, which declares the "
                        "linkage, or name a projection-only form here"
                    ),
                )
            )
        if key == "form" and pairwise and not cond.get("anchor"):
            linkages = _engine_literals("_LINKAGES")
            if not str(form.get("linkage", "") or ""):
                diags.append(
                    _diag(
                        "error",
                        "equivalence-form-linkage-required",
                        f"{cid or '(unnamed)'}: groups values under the pairwise form "
                        f"{name!r} with no `anchor`, and that form declares no `linkage` — "
                        "single and complete linkage produce different classes over identical "
                        "rows, so the engine refuses to pick one",
                        path=rel,
                        line=line,
                        detail=f"available: {', '.join(linkages) or '(undetermined)'}",
                        hint=(
                            "declare `linkage:` on the form, or `anchor:` on the condition to "
                            "compare against a fixed side instead of clustering"
                        ),
                    )
                )
        if key == "normalize" and kind == "numeric_compare":
            changed = form_aggregates()
            agg = str(cond.get("aggregate", "") or "").strip().lower()
            if agg and changed and agg not in changed:
                diags.append(
                    _diag(
                        "warning",
                        "equivalence-form-inert-aggregate",
                        f"{cid or '(unnamed)'}: `normalize` changes nothing under "
                        f"`aggregate: {agg}` — a form projects text, and this aggregate does "
                        "not read its field as text",
                        path=rel,
                        line=line,
                        detail=f"applies to: {', '.join(changed)}",
                    )
                )


#: Baseline aggregates that read a FIELD off the population's rows. ``count`` counts the rows
#: themselves and ``ratio`` counts the selected proportion of them, so neither needs one.
_BASELINE_FIELD_AGGREGATES = (
    "distinct",
    "sum",
    "min",
    "max",
    "avg",
    "median",
    "mode",
    "mode_share",
)


def _check_baseline(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Check a ``baseline`` declaration: the six ways it stops being a relative threshold.

    A ``baseline`` makes ``bound`` a MULTIPLIER of a computed population value, and it fails in
    two directions that are both silent. Where the evaluator cannot read the declaration at all
    it compares the multiplier as an ABSOLUTE bound — a `3` meant as "three times the cohort"
    tested as "more than three" — and where it reads it but cannot compute the population the
    condition is ``unknown``, so a decisive one reads as INSUFFICIENT DATA.
    """
    raw = cond.get("baseline")
    if raw is None:
        return
    readers = kinds_reading("baseline")
    if not isinstance(raw, dict):
        diags.append(
            _diag(
                "error",
                "baseline-shape",
                f"{cid or '(unnamed)'}: `baseline` is {type(raw).__name__}, not a mapping, so "
                "the evaluator ignores it and compares `bound` as an absolute threshold",
                path=rel,
                line=line,
                hint="declare `baseline:` as a mapping with at least `source` and `aggregate`",
            )
        )
        return
    if readers and kind not in readers:
        diags.append(
            _diag(
                "error",
                "baseline-unread-kind",
                f"{cid or '(unnamed)'}: `baseline` is not read by {kind!r}, so `bound` is "
                "compared as an absolute threshold while the declaration says it is a multiple "
                "of a population",
                path=rel,
                line=line,
                detail=f"read by: {', '.join(sorted(readers))}",
            )
        )
        return
    if not str(raw.get("source", "") or "").strip():
        diags.append(
            _diag(
                "error",
                "baseline-source",
                f"{cid or '(unnamed)'}: the `baseline` names no `source`, so there is no "
                "population to take a threshold off and the comparison never runs",
                path=rel,
                line=line,
                hint="name one of the ruleset's declared sources",
            )
        )
    aggregates = _engine_literals("_AGGREGATES")
    agg = str(raw.get("aggregate", "") or "").strip().lower()
    if aggregates and agg not in aggregates:
        diags.append(
            _diag(
                "error",
                "baseline-aggregate",
                f"{cid or '(unnamed)'}: the `baseline` aggregate is "
                f"{agg or '(missing)'!r}, which the evaluator does not compute, so the "
                "threshold is unreadable and the comparison never runs",
                path=rel,
                line=line,
                detail=f"available: {', '.join(sorted(aggregates))}",
            )
        )
    elif (
        agg in _BASELINE_FIELD_AGGREGATES
        and not str(raw.get("field", "") or "").strip()
    ):
        diags.append(
            _diag(
                "error",
                "baseline-field",
                f"{cid or '(unnamed)'}: the `baseline` takes the {agg} of no `field`, and "
                "there is nothing on the population's rows to aggregate",
                path=rel,
                line=line,
                hint="declare `field:` on the baseline, or count rows with `aggregate: count`",
            )
        )
    elif agg == "ratio" and not raw.get("where"):
        diags.append(
            _diag(
                "error",
                "baseline-ratio-unfiltered",
                f"{cid or '(unnamed)'}: the `baseline` is a ratio with no `where`, so every "
                "row is selected and the population value is 1 — the declared multiple is "
                "then the absolute bound, which is what declaring no baseline already does",
                path=rel,
                line=line,
                hint="declare `where:` naming the sub-population the proportion is of",
            )
        )
    if str(raw.get("per", "") or "").strip():
        inner = str(raw.get("per_aggregate", "") or "").strip().lower()
        if aggregates and (inner not in aggregates or inner == "ratio"):
            diags.append(
                _diag(
                    "error",
                    "baseline-per-aggregate",
                    f"{cid or '(unnamed)'}: the `baseline` reduces per "
                    f"{raw.get('per')!r} with `per_aggregate` {inner or '(missing)'!r}, "
                    "which yields no per-member value to combine, so the threshold is "
                    "unreadable and the comparison never runs",
                    path=rel,
                    line=line,
                    detail=(
                        "available: "
                        + ", ".join(sorted(a for a in aggregates if a != "ratio"))
                    ),
                )
            )


def _check_event_order(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Check an ``event_order``: the declaration IS the check, and every gap is silent.

    The evaluator defaults neither the relation nor the quantifier — an ordering condition that
    declares neither is not a loose one, it is one that never compares — so an unreadable
    declaration returns ``unknown`` and a decisive one reads as INSUFFICIENT DATA. A leftover
    ``max`` is the opposite failure: the condition runs, correctly, while a bound the pack
    believes it declared is ignored.
    """
    if kind != "event_order":
        return
    relations = order_relations()
    relation = str(cond.get("relation", "") or "").strip().lower()
    if relations and relation not in relations:
        diags.append(
            _diag(
                "error",
                "event-order-relation",
                f"{cid or '(unnamed)'}: the `relation` is {relation or '(missing)'!r}, which "
                "the evaluator cannot read, so the two timestamps are never compared",
                path=rel,
                line=line,
                detail=f"available: {', '.join(relations)}",
                hint="the relation is the required position of `end` relative to `start`; the "
                "`not_` forms admit simultaneity, which a day-granular column produces",
            )
        )
    quantifiers = order_quantifiers()
    quant = str(cond.get("quantifier", "") or "").strip().lower()
    if quantifiers and quant not in quantifiers:
        diags.append(
            _diag(
                "error",
                "event-order-quantifier",
                f"{cid or '(unnamed)'}: the `quantifier` is {quant or '(missing)'!r}, which "
                "the evaluator cannot read, so the two timestamps are never compared",
                path=rel,
                line=line,
                detail=f"available: {', '.join(quantifiers)}",
                hint="a side may carry many timestamps, and `every` and `any` answer different "
                "questions over one row set, so there is no default to fall back to",
            )
        )
    raw_tol = str(cond.get("tolerance", "") or "").strip()
    if raw_tol and not _WINDOW_RE.match(raw_tol):
        diags.append(
            _diag(
                "error",
                "event-order-tolerance",
                f"{cid or '(unnamed)'}: `tolerance: {cond.get('tolerance')!r}` is not a window "
                "the evaluator can read, so the ordering is never compared",
                path=rel,
                line=line,
                hint="declare a window in the engine's grammar, e.g. `tolerance: 90m` / `4h` / "
                "`2d`; an absent tolerance is exact, not a small one",
            )
        )
    for name in ("start", "end"):
        side = cond.get(name)
        side = side if isinstance(side, dict) else {}
        missing = [
            k for k in ("source", "field") if not str(side.get(k, "") or "").strip()
        ]
        if missing:
            diags.append(
                _diag(
                    "error",
                    "event-order-side",
                    f"{cid or '(unnamed)'}: the `{name}` side declares no "
                    f"{' and no '.join(f'`{k}`' for k in missing)}, so it yields no timestamp "
                    "and the condition can only report `unknown`",
                    path=rel,
                    line=line,
                    hint="each side names a `source` from the ruleset and the `field` holding "
                    "its timestamp",
                )
            )
    if cond.get("max") is not None and kind not in kinds_reading("max"):
        diags.append(
            _diag(
                "error",
                "event-order-max",
                f"{cid or '(unnamed)'}: `max: {cond.get('max')!r}` is not read by "
                f"{kind!r} — this check bounds the ORDER of the two events and not the "
                "interval between them, so the declared bound is silently ignored",
                path=rel,
                line=line,
                hint="drop it, or keep the magnitude bound as a second condition of kind "
                "`time_gap` over the same two sides",
            )
        )


def _check_composite(
    cond: Dict[str, Any],
    cid: str,
    kind: str,
    forms: Dict[str, Dict[str, Any]],
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
    depth: int = 0,
) -> None:
    """Check a composite's ``children``: how many, what they may declare, how deep they nest.

    The parent produces one finding, so a child declaring a weighting key is inert — and an
    inert `decisive` is the difference between a conclusive verdict and an advisory one. A
    single child is not a combination and reads as a typo for the child's own kind.
    """
    kinds = composite_kinds()
    if not kinds or kind not in kinds:
        return
    children = [c for c in (cond.get("children") or []) if isinstance(c, dict)]
    if len(children) < 2:
        diags.append(
            _diag(
                "error",
                "composite-children",
                f"{cid or '(unnamed)'}: a {kind!r} declares {len(children)} child "
                "condition(s); it combines two or more or it is not a combination",
                path=rel,
                line=line,
                hint=(
                    "with no children the check can only report `unknown`; with one, declare "
                    "that child's kind directly"
                ),
            )
        )
    bound = max_composite_depth()
    if bound and depth >= bound:
        diags.append(
            _diag(
                "error",
                "composite-too-deep",
                f"{cid or '(unnamed)'}: composites nest at most {bound} deep and this one "
                "goes further, so the evaluator stops and reports `unknown`",
                path=rel,
                line=line,
            )
        )
        return
    known = condition_kinds()
    for child in children:
        child_id = str(child.get("id", "") or "") or f"a child of {cid or kind}"
        child_kind = str(child.get("kind", "") or "")
        if known and child_kind not in known:
            diags.append(
                _diag(
                    "error",
                    "unknown-condition-kind",
                    f"{child_id}: kind {child_kind or '(missing)'!r} is not one the "
                    "evaluator dispatches, so this member of the combination is never "
                    "evaluated and the parent can only report `unknown`",
                    path=rel,
                    line=line,
                    detail=f"available: {', '.join(sorted(known))}",
                    hint=(
                        "an unresolved `use:` leaves a child with no kind; check the "
                        "namespace and check id"
                    ),
                )
            )
        declared = [k for k in _WEIGHTING_KEYS if k in child]
        if declared:
            diags.append(
                _diag(
                    "error",
                    "composite-child-weighting",
                    f"{child_id}: a composite's child declares {', '.join(declared)}, which "
                    "only the parent is read for — the declaration changes nothing",
                    path=rel,
                    line=line,
                    hint=f"move it onto {cid or kind!r}, which owns this check's weight",
                )
            )
        _check_pair_by(child, child_id, child_kind, rel, line, diags)
        _check_cohort_subject_rows(child, child_id, child_kind, rel, line, diags)
        _check_numeric_compare(child, child_id, child_kind, rel, line, diags)
        _check_baseline(child, child_id, child_kind, rel, line, diags)
        _check_event_order(child, child_id, child_kind, rel, line, diags)
        _check_condition_bound(child, cid=child_id, rel=rel, line=line, diags=diags)
        _check_value_equivalence(child, child_id, child_kind, rel, line, diags)
        _check_condition_form(child, child_id, child_kind, forms, rel, line, diags)
        _check_composite(
            child, child_id, child_kind, forms, rel, line, diags, depth + 1
        )


def _check_indicator_label(
    cond: Dict[str, Any],
    raw_cond: Dict[str, Any],
    *,
    cid: str,
    rel: str,
    line: int,
    diags: List[Dict[str, Any]],
) -> None:
    """Warn when a fraud indicator's label reads as a requirement rather than a finding.

    An exclusion's finding routes around its label (a fail negates a requirement). An
    indicator's label is printed verbatim because a fail affirms it. A requirement-phrased
    label under indicator polarity announces its finding by stating the opposite.

    Extra detail is reported only for imported conditions: a shared check's label is written
    for the polarity the library assumed, and importing across a polarity requires overriding
    it.
    """
    if not _label_echoes_requirement(cond.get("label"), cond.get("expected")):
        return
    imported = bool(raw_cond.get("use")) and "label" not in raw_cond
    diags.append(
        _diag(
            "warning",
            "label-polarity-unaffirmed",
            f"{cid or '(unnamed)'}: this is a fraud indicator, so a FAIL prints the label "
            "as the finding — but the label restates the requirement, which says the "
            "opposite of what was found",
            path=rel,
            line=line,
            detail=(
                f"label: {str(cond.get('label', ''))!r}; "
                f"expected: {str(cond.get('expected', ''))!r}"
                + (
                    f"; inherited unchanged from `use: {raw_cond.get('use')}`"
                    if imported
                    else ""
                )
            ),
            hint=(
                "override `label` so a FAIL affirms it"
                if imported
                else "reword `label` to state the finding, not the requirement"
            ),
        )
    )


def _render(result: Dict[str, Any]) -> List[str]:
    """The diagnostics of one pack as lines, most severe first.

    Severity leads each line. Counts are printed even when nothing fired: a checker that
    reports nothing must not read like a pack that passed.
    """
    order = {"error": 0, "warning": 1, "info": 2}
    diags = sorted(
        result.get("diagnostics") or [],
        key=lambda d: (
            order.get(str(d.get("severity")), 3),
            str(d.get("path")),
            int(d.get("line") or 0),
        ),
    )
    lines = [f"{result.get('pack')}:"]
    for d in diags:
        where = str(d.get("path") or "")
        if where and int(d.get("line") or 0):
            where += f":{int(d['line'])}"
        lines.append(
            f"  [{str(d.get('severity')).upper():<7}] {d.get('code')}"
            + (f"  {where}" if where else "")
            + f"\n            {d.get('message')}"
        )
        for label, key in (("detail", "detail"), ("hint", "hint")):
            if d.get(key):
                lines.append(f"            {label}: {d[key]}")
    counts = result.get("counts") or {}
    lines.append(
        "  checked: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    lines.append(
        f"  {result.get('errors')} error(s), {result.get('warnings')} warning(s), "
        f"{result.get('infos')} info(s)" + ("" if result.get("ok") else "  <- FAILS")
    )
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m src.knowledge.pack_validate <dir> [<dir>...]``.

    Takes directories: the checked-in template lives outside the packs root and must be
    lintable too. Exits non-zero only on an error, mirroring ``ok``: warnings never block
    because several are claims about prose that no mechanical check can settle.

    A directory that cannot be read is its own non-zero exit so a mistyped path does not
    read like a pack with no findings.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: python -m src.knowledge.pack_validate <pack-dir> [<pack-dir>...]",
            file=sys.stderr,
        )
        return 2
    worst = 0
    for i, raw in enumerate(args):
        d = Path(raw)
        if not d.is_dir():
            print(f"{raw}: not a directory", file=sys.stderr)
            worst = max(worst, 2)
            continue
        try:
            result = validate_pack(d)
        except Exception as exc:  # noqa: BLE001 - report, never traceback at a CLI
            print(
                f"{raw}: could not be validated: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            worst = max(worst, 2)
            continue
        if i:
            print("")
        print("\n".join(_render(result)))
        if not result.get("ok"):
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    sys.exit(main())
