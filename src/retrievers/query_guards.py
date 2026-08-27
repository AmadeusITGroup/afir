"""Deterministic guards applied to an LLM-generated backend query.

All rewrites are conservative: only unambiguous shapes are touched; anything else is left
as generated and logged. Two query families are covered:

* SQL-like dialects (SQL, ``ES|QL``): :func:`strip_evidence_predicates`,
  :func:`enforce_conjunction`, :func:`enforce_value_tuples`,
  :func:`strip_fabricated_predicates`, :func:`enforce_partition_bounds`,
  :func:`enforce_epoch_window`;
* Elasticsearch Query DSL: :func:`strip_evidence_filters_dsl`,
  :func:`enforce_conjunction_dsl`, :func:`enforce_value_tuples_dsl`,
  :func:`strip_fabricated_filters_dsl`, :func:`enforce_partition_bounds_dsl`,
  :func:`enforce_epoch_window_dsl`.
"""

import copy
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

# Comparison operators a generated predicate may use. `==` is `ES|QL`, `=` is SQL.
_OPERATORS = r"(?:==|=|!=|<>|\bIS\b(?:\s+NOT)?|\bIN\b|\bLIKE\b|\bRLIKE\b)"
# Right-hand side: a parenthesised list, a quoted string, or a bare word (true/false/number).
_LITERAL = r"(?:\([^)]*\)|'[^']*'|\"[^\"]*\"|[\w.]+)"
# Only the quoted forms. Guards that test for a keyword must blank quoted literals before
# testing, because `_LITERAL`'s bare-word alternative would also blank every keyword.
_QUOTED_LITERAL = r"(?:'[^']*'|\"[^\"]*\")"


def _blank_quoted(text: str) -> str:
    """``text`` with every quoted literal replaced by spaces of the same length.

    Length-preserving so an offset into the result still indexes the original.
    """
    return re.sub(_QUOTED_LITERAL, lambda m: " " * len(m.group(0)), text)


def leaf_of(field: str) -> str:
    """Last segment of a dotted/backticked field path (``a.b.`c``` -> ``c``).

    Matching is on the leaf because the generated query may reference the field bare,
    dotted, backticked, or via a table alias.
    """
    return str(field).split(".")[-1].strip('`"[] ')


def _predicate_key(predicate: str) -> tuple:
    """``(leaf, whitespace-normalised predicate)``: the dedup key for one emitted predicate.

    Both dedup sites (disjuncts within a group and scopes after the lift) must agree on what
    "the same predicate" means; one function ensures they do.
    """
    return (
        leaf_of(predicate.split("=")[0]),
        re.sub(r"\s+", "", predicate).lower(),
    )


def _dedupe_predicates(predicates: List[str]) -> List[str]:
    """``predicates`` with later repeats of the same ``_predicate_key`` removed, order kept."""
    seen: set = set()
    out: List[str] = []
    for predicate in predicates:
        key = _predicate_key(predicate)
        if key in seen:
            continue
        seen.add(key)
        out.append(predicate)
    return out


def _predicate_pattern(field: str) -> str:
    """Regex for ``<optional path/alias><leaf> <op> <literal>``."""
    return (
        r"(?:`?\w+`?\.)*`?"
        + re.escape(leaf_of(field))
        + r"`?\s*"
        + _OPERATORS
        + r"\s*"
        + _LITERAL
    )


def _drop_evidence_disjuncts(
    text: str, fields: List[str], source_name: str = "?"
) -> str:
    """Drop an OR-ed arm that filters on a ``never_filter`` field, where another arm survives.

    Removing a never_filter predicate that is AND-ed is a widening and must be handled
    carefully by the caller. Removing one that is OR-ed is a narrowing and is always safe,
    because the arm satisfies the whole group on its own (every row carrying that value passes),
    making any subject-scoping beside it decorative.

    Three refusals: a compound arm (names other columns too, so dropping it could lose identity
    predicates); all arms dropped (the group must not be emptied); any ``NOT`` outside a quoted
    literal (under negation dropping a disjunct excludes less, which is forbidden).
    """
    patterns = [
        (field, re.compile(r"^\s*" + _predicate_pattern(field) + r"\s*$", re.IGNORECASE))
        for field in fields
        if field
    ]
    if not patterns or not text:
        return text
    if re.search(r"\bNOT\b", _blank_quoted(text), re.IGNORECASE):
        return text

    def _rewrite(body: str, whole: str) -> str:
        arms = _split_top_level_or(body)
        if arms is None:  # a type guard; the walker already split this body
            return whole
        keep: List[str] = []
        dropped: List[str] = []
        for arm in arms:
            field = next(
                (f for f, pattern in patterns if pattern.match(arm)),
                None,
            )
            if field is None:
                keep.append(arm.strip())
            else:
                dropped.append(arm.strip())
        if not dropped or not keep:
            return whole
        logger.warning(
            "Source '%s': dropped %d OR-ed arm(s) on evidence field(s) the pack declares must "
            "never be FILTERED — such an arm is true of every row carrying that value, so it "
            "satisfies the group ALONE and the %d arm(s) scoping it select nothing. Dropped: "
            "%s. Kept verbatim: %s.",
            source_name,
            len(dropped),
            len(keep),
            "; ".join(dropped),
            "; ".join(keep),
        )
        return "(" + " OR ".join(keep) + ")"

    return _rewrite_or_groups(text, _rewrite)


def strip_evidence_predicates(
    text: str,
    never_filter: Iterable[str],
    source_name: str = "?",
    pipe_stages: bool = False,
) -> str:
    """Remove predicates on ``never_filter`` fields from a textual query's WHERE clause.

    Handled shapes: ``... AND <field> <op> <literal>``; a leading WHERE predicate followed by
    AND; with ``pipe_stages`` (``ES|QL``), a whole ``| WHERE <field> <op> <literal>`` stage; and
    a whole disjunct of an OR-group via :func:`_drop_evidence_disjuncts`, which runs first.
    Any other shape is left as generated and logged. Returns ``text`` unchanged when
    ``never_filter`` is empty.
    """
    fields = [f for f in (never_filter or []) if f]
    if not fields or not text:
        return text
    out = _drop_evidence_disjuncts(text, fields, source_name)
    for field in fields:
        pred = _predicate_pattern(field)
        patterns: List[str] = []
        if pipe_stages:
            # A pipe stage that is nothing but the forbidden predicate: drop the stage.
            patterns.append(r"\|\s*WHERE\s+" + pred + r"\s*(?=\||$)")
        patterns += [
            r"\s+AND\s+" + pred,  # ... AND <pred>
            r"(?<=\bWHERE\s)" + pred + r"\s+AND\s+",  # WHERE <pred> AND ...
        ]
        for pattern in patterns:
            out, n = re.subn(pattern, " ", out, flags=re.IGNORECASE)
            if n:
                logger.warning(
                    "Source '%s': stripped %d filter(s) on evidence field '%s' from the "
                    "generated query — that field must be RETURNED, not filtered (a "
                    "predicate on it deletes the rows that answer the question).",
                    source_name,
                    n,
                    field,
                )
        # Still referenced in a WHERE we could not safely rewrite (e.g. inside an
        # OR-group): say so loudly rather than guess at the boolean structure.
        tail = re.split(r"\bWHERE\b", out, maxsplit=1, flags=re.IGNORECASE)
        if len(tail) > 1 and re.search(pred, tail[1], re.IGNORECASE):
            logger.warning(
                "Source '%s': evidence field '%s' is still filtered in the generated "
                "query in a shape too complex to rewrite safely; rows that would answer "
                "the check may be missing.",
                source_name,
                field,
            )
    return out


def synonym_families(synonym_fields: Any) -> List[List[str]]:
    """Normalise ``identity_synonyms`` to a list of families: OR inside, AND between.

    The declaration accepts a flat list (one family of interchangeable columns) or a list of
    lists (several independent identifiers, each spread over several columns). One function
    so the textual rewrite, the DSL rewrite and any prompt hint can never disagree about
    where a family boundary is; a boundary read one way here and another way there is
    indistinguishable from no boundary at all.
    """
    items = [f for f in (synonym_fields or []) if f]
    if not items:
        return []
    if all(isinstance(f, (list, tuple)) for f in items):
        return [[str(c) for c in fam if c] for fam in items if any(fam)]
    # A flat list, or a mix. A mix is authored (not generated), so read every bare entry as
    # its own member of the single family rather than guessing at intent.
    flat: List[str] = []
    for f in items:
        flat.extend(str(c) for c in f if c) if isinstance(f, (list, tuple)) else flat.append(str(f))
    return [flat] if flat else []


def _group_by_family(
    pieces: List[tuple], fam_leaves: List[List[str]], source_name: str = "?"
) -> tuple:
    """Group ``(field, clause)`` pairs into AND-conjuncts by declared family.

    Returns ``(groups, saw_partial)``. Shared by the textual and DSL rewrites so both routes
    apply the same family-boundary rules.

    A family is AND-ed only when the query constrained every declared member. An incomplete
    family is not AND-ed and its clauses are discarded: AND-ing a partial family is a narrower
    filter than declared and deletes every row of the shapes the missing columns would have
    matched.
    """
    groups: List[List[Any]] = []
    claimed: set = set()
    saw_partial = False
    present = {leaf for leaf, _ in pieces}
    for family in fam_leaves:
        members = [c for leaf, c in pieces if leaf in set(family)]
        if not members:
            continue
        covered = present & set(family)
        if covered == set(family):
            groups.append(members)
            claimed.update(id(c) for c in members)
            continue
        saw_partial = True
        claimed.update(id(c) for c in members)
        logger.warning(
            "Source '%s': identity family %s was constrained on %d of its %d columns, so it "
            "is NOT enforced and its clause(s) are dropped from the rewrite — AND-ing a "
            "partial family filters on fewer columns than declared and deletes every row of "
            "the shapes the missing columns would have matched.",
            source_name,
            "/".join(family),
            len(covered),
            len(set(family)),
        )
    return groups, saw_partial


def _split_top_level_or(body: str) -> Optional[List[str]]:
    """Split ``body`` on top-level ORs (paren depth zero), or ``None`` if input is unbalanced.

    A lookahead regex cannot do this: it splits the first OR it finds regardless of nesting
    depth, which inverts the result on grouped arms. Depth counting is required. Quotes are
    tracked because a literal may contain a parenthesis.
    """
    return _split_top_level(body, "OR")


def _split_top_level(body: str, operator: str) -> Optional[List[str]]:
    """Split ``body`` on ``operator`` at paren depth zero, or ``None`` if unbalanced.

    The depth- and quote-aware reader behind :func:`_split_top_level_or`, parametrised to
    support AND splits (used by :func:`_is_constrained_conjunctively`). Callers that need
    offsets for in-place splicing use :func:`_split_top_level_spans`; the two share the same
    underlying scan so they cannot disagree about boundaries.
    """
    spans = _split_top_level_spans(body, operator)
    if spans is None:
        return None
    return [body[start:end] for start, end in spans]


def _split_top_level_spans(body: str, operator: str) -> Optional[List[tuple]]:
    """``(start, end)`` of every ``operator``-separated part of ``body`` at paren depth zero.

    Offset form of :func:`_split_top_level`, and the only underlying scanner. Offsets are
    required for in-place splicing: ``str.find`` would match the first occurrence of the same
    predicate rather than the one at the position that was read. Returns ``None`` on unbalanced
    input.
    """
    spans: List[tuple] = []
    depth = 0
    quote = ""
    start = 0
    i = 0
    n = len(body)
    width = len(operator)
    upper = operator.upper()
    while i < n:
        ch = body[i]
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
        elif (
            depth == 0
            and body[i : i + width].upper() == upper
            and (i == 0 or not (body[i - 1].isalnum() or body[i - 1] == "_"))
            and (
                i + width >= n
                or not (body[i + width].isalnum() or body[i + width] == "_")
            )
        ):
            spans.append((start, i))
            i += width
            start = i
            continue
        i += 1
    if depth != 0 or quote:
        return None
    spans.append((start, len(body)))
    return spans


def _rewrite_or_groups(text: str, rewrite) -> str:
    """Offer every parenthesised group containing a top-level OR to ``rewrite``.

    ``rewrite(body, whole)`` returns either a replacement or ``whole`` unchanged. Groups are
    offered outermost first: the outermost OR decides what the predicate means, and offering
    an inner group first would let a local rewrite change an arm the outer reader is about to
    classify. A rewritten group is not re-entered. Unbalanced text yields no groups.
    """
    spans = _paren_spans(text)
    if not spans:
        return text
    # Decide on every span against the original text, then splice. Deciding and splicing in one
    # pass would read offsets into a string that earlier splices have already shifted.
    accepted: List[tuple] = []
    for start, end in spans:
        if any(s <= start and end <= e for s, e, _ in accepted):
            continue  # inside a group already rewritten; skip
        whole = text[start:end]
        parts = _split_top_level_or(whole[1:-1])
        if parts is None or len(parts) < 2:
            continue  # no top-level OR here; the arms are offered on their own turn
        new = rewrite(whole[1:-1], whole)
        if new != whole:
            accepted.append((start, end, new))
    out = text
    # Highest offset first, so every remaining offset still points where it did in `text`.
    for start, end, new in sorted(accepted, key=lambda a: -a[0]):
        out = out[:start] + new + out[end:]
    return out


def _paren_spans(text: str) -> List[tuple]:
    """``(start, end)`` of every balanced parenthesised group, outermost first.

    Quote-aware for the same reason :func:`_split_top_level_or` is: an unpaired parenthesis
    inside a string literal would shift the depth for everything after it. Unbalanced input
    yields whatever balanced groups were closed before the imbalance, which is the conservative
    answer; a caller can only ever leave such text alone.
    """
    spans: List[tuple] = []
    stack: List[int] = []
    quote = ""
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            spans.append((stack.pop(), i + 1))
    # `spans` closes innermost-first; sorting by start (then by widest) puts the outermost
    # group of each nesting ahead of the arms it contains, which is the order the caller needs.
    return sorted(spans, key=lambda s: (s[0], -s[1]))


def _plain_subgroup_operands(part: str, operand: str) -> Optional[List]:
    """Every ``<field> <op> <literal>`` operand of ``part``, or ``None`` if the arm says more.

    Returns ``[(leaf, text), ...]`` so the caller can both decide what the arm constrains and
    re-use the predicates verbatim. Returns ``None`` whenever the text is not fully accounted
    for by comparisons, parentheses and boolean words: the check is on the residue rather than
    an allowlist, so an unrecognised shape declines rather than fails open.
    """
    inner = part.strip()
    while inner.startswith("(") and inner.endswith(")"):
        stripped = inner[1:-1]
        if _split_top_level_or(stripped) is None:
            break  # the outer parens are not a matched pair around this text
        inner = stripped.strip()
    found = [
        (m.group(1), m.group(0).strip())
        for m in re.finditer(operand, inner, re.IGNORECASE)
        if m.group(1)
    ]
    if not found:
        return None
    residue = re.sub(operand, " ", inner, flags=re.IGNORECASE)
    residue = re.sub(r"\b(AND|OR|NOT)\b|[()\s]", " ", residue, flags=re.IGNORECASE)
    if residue.strip():
        return None
    return found


def _wrapped_arm_leaf(part: str, operand: str) -> Optional[str]:
    """Column an OR-ed arm constrains when its syntax cannot be decomposed.

    Called by :func:`enforce_identity_scope` after :func:`_plain_subgroup_operands` declines.
    Returns the column only when exactly one comparison is present (two or more is ambiguous);
    the arm is re-emitted verbatim so the only effect is which family slot it occupies.
    A negation anywhere outside a literal is refused: an excluding arm does not identify
    the subject.
    """
    hits = [
        m.group(1) for m in re.finditer(operand, part, re.IGNORECASE) if m.group(1)
    ]
    if len(hits) != 1:
        return None
    if re.search(r"\bNOT\b|<>|!=", _blank_quoted(part), re.IGNORECASE):
        return None
    return hits[0]


def resolve_identity_fields(
    declared: Any, field_map: Dict[str, str], source_name: str = "?"
) -> Any:
    """Resolve an identity declaration to this source's real columns, keeping its nesting.

    A name is an entity type if it is a key of ``field_map``; otherwise it is treated as a
    concrete column. Testing for a dot instead would silently drop top-level columns on flat
    backends. A name that resolves to neither a mapped type nor a real column is harmless: the
    guard only moves clauses the generator already wrote. Shared by all retrievers that enforce
    the shape so the SQL and DSL routes cannot disagree about which columns a family contains.
    Families (a list of lists) keep their grouping; a flat list stays flat.
    """
    items = [d for d in (declared or []) if d]
    if items and all(isinstance(d, (list, tuple)) for d in items):
        out_families = []
        for family in items:
            resolved = resolve_identity_fields(family, field_map, source_name)
            if resolved:
                out_families.append(resolved)
        return out_families
    out: List[str] = []
    for item in items:
        field = (field_map or {}).get(item) or str(item)
        if field not in out:
            out.append(field)
    return out


def enforce_identity_scope(
    text: str,
    scope_fields: List[str],
    synonym_fields: Optional[Any] = None,
    source_name: str = "?",
) -> str:
    """AND an event log's scope fields onto its OR-group of synonym fields.

    Synonyms are OR-ed (one column may be null); scopes are AND-ed separate facts. Only an
    existing top-level OR-group whose disjuncts are all on declared fields is touched; scope
    predicates inside it are lifted out and AND-ed after the group. Any other shape is left
    as generated and logged. Nothing is invented: a scope the query never constrained is not
    added.

    ``synonym_fields`` may declare several families (see :func:`synonym_families`); they are
    OR-ed inside and AND-ed between. A family is AND-ed only when constrained completely.
    """
    if not text:
        return text
    families = synonym_families(synonym_fields)
    if not (scope_fields or []) and len(families) < 2:
        return text
    scope_leaves = [leaf_of(f) for f in (scope_fields or []) if f]
    fam_leaves = [[leaf_of(f) for f in fam] for fam in families]
    syn_leaves = [leaf for fam in fam_leaves for leaf in fam]
    if not scope_leaves and len(families) < 2:
        return text
    known = set(scope_leaves) | set(syn_leaves)

    # The leaf is the last dotted segment, so the path prefix must be non-greedy or it
    # swallows all but the final character of the name we are trying to capture.
    operand = (
        r"(?:`?[\w.]+?`?\.)?`?(\w+)`?\s*(?:==|=|\bIN\b)\s*" + _LITERAL
    )

    def _rewrite(body: str, whole: str) -> str:
        parts = _split_top_level_or(body)
        if parts is None or len(parts) < 2:
            return whole
        pieces = []
        lifted: List[str] = []
        dropped: List = []
        for part in parts:
            part = part.strip()
            m = re.fullmatch(operand, part, re.IGNORECASE)
            if not m:
                # Nested arm: classify by whether it constrains a synonym column.
                # Synonym column: flatten its operands into pieces.
                # Scope-only: lift declared predicates; ambiguous or undeclared: leave whole group.
                inner = _plain_subgroup_operands(part, operand)
                if inner is None:
                    # Not decomposable; try to attribute by single comparison only.
                    # A synonym-column arm is re-emitted verbatim; a scope arm or ambiguous
                    # arm declines the group.
                    wrapped = _wrapped_arm_leaf(part, operand)
                    if wrapped is None or wrapped not in set(syn_leaves):
                        return whole  # not a shape we can read; leave alone
                    pieces.append((wrapped, part))
                    continue
                if any(leaf in set(syn_leaves) for leaf, _ in inner):
                    # Actor's identity arm. Every column must be declared; undeclared columns
                    # have no family to belong to after re-grouping.
                    if any(leaf not in known for leaf, _ in inner):
                        return whole
                    for leaf, predicate in inner:
                        pieces.append((leaf, predicate))
                    continue
                if not any(leaf in set(scope_leaves) for leaf, _ in inner):
                    # Neither synonym nor declared scope: unclassifiable, leave as generated.
                    return whole
                # Scope arm: lift only declared scope predicates; an undeclared column
                # paired with a scope is left in the arm.
                keep = [t for leaf, t in inner if leaf in set(scope_leaves)]
                lifted.extend(keep)
                # Recorded, not logged yet: every path below may still decline and return the
                # query unchanged, so a warning must not fire before the drop is committed.
                dropped.append(
                    (
                        ", ".join(leaf for leaf, _ in inner) or "?",
                        ", ".join(leaf_of(t.split("=")[0]) for t in keep) or "nothing",
                    )
                )
                continue
            leaf = m.group(1)
            if leaf not in known:
                return whole  # an unrelated disjunct; do not restructure
            pieces.append((leaf, part))
        # One predicate per (column, literal), keeping the first spelling. Arms of one group
        # routinely repeat a predicate when a correct conjunction is OR-ed with its bare arms.
        seen: set = set()
        deduped = []
        for leaf, part in pieces:
            key = (leaf, _predicate_key(part)[1])
            if key in seen:
                continue
            seen.add(key)
            deduped.append((leaf, part))
        pieces = deduped
        if dropped and not pieces:
            # Every arm was dropped; lifting scopes alone would produce a whole-population
            # query. Lifting is only sound alongside a surviving identity arm.
            return whole
        # Deduped after the lift: a scope can arrive by two routes (bare disjunct read into
        # pieces, and lifted from a dropped arm) and must not be AND-ed on twice.
        scopes = _dedupe_predicates(
            [p for leaf, p in pieces if leaf in set(scope_leaves)] + lifted
        )
        grouped, partial = _group_by_family(
            [(leaf, p) for leaf, p in pieces if leaf not in set(scope_leaves)],
            fam_leaves,
            source_name,
        )

        def _report_drops() -> None:
            for arm, kept in dropped:
                logger.warning(
                    "Source '%s': dropped the OR-ed arm (%s) from the identity group, "
                    "keeping %s as a conjunct — an arm naming no column that can hold the "
                    "actor's own identifier asks for every row sharing that scope, and under "
                    "the row cap the subject's own rows need not be in what comes back at "
                    "all. They may not exist, which is a FINDING this shape hides.",
                    source_name,
                    arm,
                    kept,
                )

        # One family, no scope and nothing dropped is what a plain OR-group already means.
        if not scopes and len(grouped) < 2 and not partial and not dropped:
            return whole
        if not grouped and partial:
            # Every family in the group was partial, so nothing left would identify the
            # subject. Enforcing the scopes alone is the whole-unit query this guard exists
            # to prevent; leave it as generated rather than narrow to the wrong thing.
            return whole
        if not grouped:
            # Every disjunct is a separate scope; OR-ing them is the "everyone in the
            # same unit" bug with no synonym group involved. AND them.
            _report_drops()
            return "(" + " AND ".join(scopes) + ")"
        conjuncts = [
            ("(" + " OR ".join(g) + ")") if len(g) > 1 else g[0] for g in grouped
        ]
        rebuilt = " AND ".join(conjuncts)
        for scope in scopes:
            rebuilt += " AND " + scope
        _report_drops()
        logger.warning(
            "Source '%s': rewrote a flat identity OR-group into %s%s — OR-ing a SCOPE (a "
            "unit, a role) or a SECOND independent identifier onto the actor's own returns "
            "every other actor sharing it, which under the row cap can crowd out the "
            "subject entirely.",
            source_name,
            " AND ".join(
                "(" + " OR ".join(leaf_of(s.split("=")[0]) for s in g) + ")"
                if len(g) > 1
                else leaf_of(g[0].split("=")[0])
                for g in grouped
            ),
            (" AND " + " AND ".join(leaf_of(s.split("=")[0]) for s in scopes))
            if scopes
            else "",
        )
        return "(" + rebuilt + ")"

    out = _rewrite_or_groups(text, _rewrite)
    if out == text and re.search(r"\bOR\b", text, re.IGNORECASE):
        logger.info(
            "Source '%s': identity scope fields %s declared, but no rewritable flat "
            "OR-group was found — the query is left exactly as generated.",
            source_name,
            ", ".join(scope_leaves) or "/".join("+".join(f) for f in fam_leaves),
        )
    return out


def subject_anchor_values(
    values_by_field: Dict[str, List[str]],
    scope_fields: Any,
    synonym_fields: Any,
    source_name: str = "?",
) -> tuple:
    """Split this incident's resolved values into ``(identity, scope)`` column maps.

    Reads the same two pack declarations as :func:`enforce_identity_scope`: synonyms OR-ed,
    scopes AND-ed. A value on one synonym column is placed on every column of that family,
    because the producer populates whichever column it uses and anchoring on only one drops
    every row where the other was populated. A column that is neither a declared synonym nor
    a declared scope is ignored; the pack says which columns are which. A source declaring
    neither list yields two empty maps and the anchor is a no-op.
    """
    if not values_by_field:
        return {}, {}
    identity: Dict[str, List[str]] = {}
    scopes: Dict[str, List[str]] = {}
    for family in synonym_families(synonym_fields):
        columns = [str(f) for f in family if f]
        found: List[str] = []
        for col in columns:
            for value in values_by_field.get(col, []) or []:
                if value not in found:
                    found.append(value)
        if not found:
            continue
        for col in columns:
            identity[col] = list(found)
    for field in scope_fields or []:
        col = str(field)
        values = values_by_field.get(col) or []
        if values:
            scopes[col] = list(values)
    if not identity and scopes:
        # Scopes but no identity: this incident carries no value for any column that can hold
        # the subject's own identifier here. AND-ing the scopes alone is the whole-population
        # query the identity guard refuses to write, so nothing is anchored.
        logger.info(
            "Source '%s': the incident carries a value for the declared scope(s) %s but for "
            "no column that can hold the subject's own identifier, so no anchor is built — "
            "the scopes alone would ask about everybody sharing them.",
            source_name,
            ", ".join(leaf_of(c) for c in scopes),
        )
        return {}, {}
    return identity, scopes


def _anchor_clause(
    identity_values: Dict[str, List[str]],
    scope_values: Dict[str, List[str]],
    dialect: str = "sql",
) -> str:
    """``(id OR id2) AND scope`` from resolved ``{column: [value, ...]}`` maps, or ``''``.

    Pure renderer, so the SQL route and the DSL route share one decision about the
    output shape and differ only in how a clause is spelled.
    """
    if not identity_values:
        return ""

    def _eq(column: str, values: List[str]) -> str:
        literals = [_quote(v, "STRING", dialect) for v in values]
        op = "==" if dialect == "esql" else "="
        if len(literals) == 1:
            return f"{column} {op} {literals[0]}"
        return f"{column} IN ({', '.join(literals)})"

    arms = [_eq(col, vals) for col, vals in identity_values.items() if vals]
    if not arms:
        return ""
    clause = arms[0] if len(arms) == 1 else "(" + " OR ".join(arms) + ")"
    for col, vals in scope_values.items():
        if vals:
            clause += " AND " + _eq(col, vals)
    return clause


def _and_clause_onto(text: str, clause: str, dialect: str = "sql") -> str:
    """``text`` with ``clause`` AND-ed onto its outer filter, body parenthesised.

    Shared by the additive guards. Parenthesises the existing body so a top-level OR cannot
    be re-associated. For SQL, extends the outermost WHERE (found by depth, not by position)
    so a query carrying a subquery with its own WHERE clause is handled correctly.
    """
    if dialect == "esql":
        head, sep, rest = text.partition("|")
        return (
            f"{head.rstrip()} | WHERE {clause} {sep}{rest}"
            if sep
            else f"{text.rstrip()} | WHERE {clause}"
        )
    # The outer WHERE, found by depth, not by position. Every other guard declines on a
    # second WHERE because choosing is guesswork for them; here the outer one is unambiguous.
    end = _outer_where_end(text)
    if end is not None:
        head, rest = text[:end], text[end:]
        cut = _tail_index(rest)
        body, tail = rest[:cut].strip(), rest[cut:]
        out = f"{head} {clause} AND ({body})"
        if tail.strip():
            out = f"{out} {tail.strip()}"
        return out
    cut = _tail_index(text)
    body, tail = text[:cut].rstrip(), text[cut:]
    out = f"{body} WHERE {clause}"
    if tail.strip():
        out = f"{out} {tail.strip()}"
    return out


def _and_clause_onto_arms(
    text: str, clause: str, source_name: str, guard: str, dialect: str = "sql"
) -> Optional[str]:
    """:func:`_and_clause_onto`, applied to every arm of a top-level set operation.

    A set operation returns the union of its arms' rows, so an arm left unchanged contributes
    the rows the clause was injected to exclude. Used by all three additive guards.

    Returns ``None`` when any arm is parenthesised: there is no depth-zero WHERE to extend,
    and appending one after the closing bracket is not valid SQL. A refusal costs the clause;
    a broken statement costs the source.
    """
    if dialect != "sql":
        return _and_clause_onto(text, clause, dialect=dialect)
    arms, operators = _set_operation_arms(text)
    if len(arms) < 2:
        return _and_clause_onto(text, clause, dialect=dialect)
    if any(arm.lstrip().startswith("(") for arm in arms):
        return None
    out = _and_clause_onto(arms[0].strip(), clause, dialect=dialect)
    for operator, arm in zip(operators, arms[1:]):
        out = (
            f"{out}\n{operator.strip()}\n"
            f"{_and_clause_onto(arm.strip(), clause, dialect=dialect)}"
        )
    logger.info(
        "%s on %s: the statement is a %d-arm set operation — the clause is AND-ed onto EVERY "
        "arm, or the arms this guard skipped would contribute the very rows it exists to "
        "exclude.",
        guard,
        source_name,
        len(arms),
    )
    return out


def _sql_literal(value: Any) -> str:
    """A pack-declared constant as a SQL literal, quoted unless it is genuinely numeric.

    Quoting a number is usually harmless and occasionally is not (a comparison against a
    numeric column can fail to plan or silently cast), and leaving a string unquoted is a
    syntax error, so the two cases are told apart here rather than at each call site.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def enforce_default_filters(
    text: str,
    default_filters: Dict[str, Any],
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """AND ``default_filters`` onto the generated query as a mandatory conjunct.

    Applied to every arm of a top-level set operation so no arm escapes the pinned slice.
    Multiple entries are AND-ed; a fact with several spellings across document shapes must not
    use this key, because no row carries two spellings.

    Compose with ``never_filter``: that strips the constant from any OR-group where it satisfies
    the group alone; this pins it as a conjunct. The two keys are not interchangeable.
    """
    if not text or not default_filters:
        return text
    clause = " AND ".join(
        f"{field} = {_sql_literal(value)}" for field, value in default_filters.items()
    )
    logger.info(
        "default_filters on %s: AND-ing the pinned slice onto the query — %s",
        source_name,
        clause,
    )
    out = _and_clause_onto_arms(
        text, clause, source_name, "default_filters", dialect=dialect
    )
    if out is None:
        logger.warning(
            "default_filters on %s: the statement unions a PARENTHESISED arm, which has no "
            "outer WHERE to extend — left as generated, so the pinned slice (%s) is NOT "
            "enforced on this query.",
            source_name,
            clause,
        )
        return text
    return out


# The set-operation boundaries `enforce_default_filters` splices across. `UNION ALL` is listed
# before the bare `UNION` because alternation is ordered and the shorter form would otherwise
# match first, leaving `ALL` at the head of the next arm.
_SET_OPERATION_TOKEN = re.compile(
    r"\b(?:UNION\s+ALL|UNION\s+DISTINCT|UNION|EXCEPT\s+ALL|EXCEPT|INTERSECT\s+ALL|INTERSECT)\b",
    re.IGNORECASE,
)


def _set_operation_arms(text: str) -> "tuple[List[str], List[str]]":
    """``([arm, ...], [operator, ...])`` for a top-level ``UNION``/``EXCEPT``/``INTERSECT``.

    One arm and no operator for an ordinary statement, so the caller needs no special case.
    Depth- and quote-aware for the reason every scanner in this module is: the token appears
    inside subqueries and inside string literals, and splitting there would cut a statement in
    half. The operator text is returned rather than normalised so it can be re-emitted exactly
    as the generator wrote it; ``UNION ALL`` and ``UNION`` are different statements.
    """
    arms: List[str] = []
    operators: List[str] = []
    depth = 0
    quote = ""
    prev = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            match = _SET_OPERATION_TOKEN.match(text, i)
            if match:
                arms.append(text[prev : match.start()])
                operators.append(match.group(0))
                prev = match.end()
                i = match.end()
                continue
        i += 1
    arms.append(text[prev:])
    return arms, operators


def enforce_key_presence(
    text: str,
    key_values: "Dict[str, Dict[str, List[str]]]",
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """Add a resolved key member the generated query constrains nowhere.

    ``enforce_conjunction`` can only turn an OR it finds; a missing member needs this guard.
    Five conservatism constraints: per entity type with columns OR-ed
    (``{entity type: {column: [value, ...]}}``) so both spellings need not appear on the same
    row; already constrained conjunctively: untouched (a comparison OR-ed beside a wider arm
    binds no row); no literal invented; columns from ``field_mapping.key_presence_values``;
    all-or-nothing per member, members AND-ed.

    ``key_values`` must hold only the key's types: AND-ing a non-key entity voids the absence
    proof (see :func:`key_was_enforced`).
    """
    if not text or not key_values:
        return text
    missing = {}
    for entity, columns in key_values.items():
        columns = {c: v for c, v in (columns or {}).items() if c and v}
        if not columns:
            continue
        if _is_constrained_conjunctively(text, list(columns), dialect):
            continue
        missing[entity] = columns
    if not missing:
        return text
    clause = " AND ".join(
        filter(None, (_anchor_clause(cols, {}, dialect) for cols in missing.values()))
    )
    if not clause:
        return text
    out = _and_clause_onto_arms(text, clause, source_name, "key presence", dialect)
    if out is None:
        logger.warning(
            "Source '%s': the statement unions a PARENTHESISED arm, which has no outer WHERE "
            "to extend — left as generated, so the key member(s) (%s) are NOT enforced and an "
            "empty result from this query says nothing about this key.",
            source_name,
            clause,
        )
        return text
    logger.warning(
        "Source '%s': injected the key member(s) %s — this source declares the actor key "
        "%s, and the generated query constrained %s on no column, so it asked about a "
        "population rather than about this key. On a keyed lookup that also decides what an "
        "empty result MEANS.",
        source_name,
        "; ".join(
            f"{entity}=({' OR '.join(leaf_of(c) for c in cols)})"
            if len(cols) > 1
            else f"{entity}={leaf_of(next(iter(cols)))}"
            for entity, cols in missing.items()
        ),
        " + ".join(key_values),
        " + ".join(missing),
    )
    return out


def enforce_subject_anchor(
    text: str,
    identity_values: Dict[str, List[str]],
    scope_values: Dict[str, List[str]],
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """Add the subject's identity to a query that constrains it nowhere.

    Conservative in four ways: already anchored conjunctively (strong test, since a comparison
    OR-ed beside a wider arm binds no row) is untouched; nothing to inject means no-op;
    identity columns are OR-ed, scope AND-ed; the existing body is parenthesised before the
    AND so a top-level OR cannot be re-associated.

    ``identity_values`` / ``scope_values`` are ``{real column: [value, ...]}``, resolved from
    ``field_mapping.filter_values_by_field`` so the hint and this injection name the same
    columns.
    """
    if not text or not identity_values:
        return text
    if _is_constrained_conjunctively(text, identity_values, dialect):
        logger.info(
            "Source '%s': every row the generated query can return is bound to the incident's "
            "identity (%s), so the subject anchor is not injected — the generator's own "
            "scoping stands, including a tighter one.",
            source_name,
            ", ".join(leaf_of(c) for c in identity_values),
        )
        return text
    clause = _anchor_clause(identity_values, scope_values or {}, dialect)
    if not clause:
        return text

    out = _and_clause_onto_arms(text, clause, source_name, "subject anchor", dialect)
    if out is None:
        logger.warning(
            "Source '%s': the statement unions a PARENTHESISED arm, which has no outer WHERE "
            "to extend — left as generated, so the subject anchor (%s) is NOT enforced and the "
            "query asks about a population.",
            source_name,
            clause,
        )
        return text
    logger.warning(
        "Source '%s': injected the subject anchor %s — the generated query constrained the "
        "incident's identity on no column, so it asked about a population rather than about "
        "this subject. Under the row cap the subject's own rows need not have been in the "
        "page at all.",
        source_name,
        " AND ".join(
            [
                "("
                + " OR ".join(leaf_of(c) for c in identity_values)
                + ")"
                if len(identity_values) > 1
                else leaf_of(next(iter(identity_values)))
            ]
            + [leaf_of(c) for c in (scope_values or {})]
        ),
    )
    return out


def _or_group_widens_one_key_field(group: str, key_leaves: set) -> bool:
    """Is this ``OR`` group a widening of a single key field's own equality?

    Read only by :func:`key_was_enforced`, and only to interpret an empty result, which is
    what makes a disjunct admissible at all: if ``sign = 'X' OR sign LIKE 'X%'`` returns no
    row then ``sign = 'X'`` returned no row either, because the second query asks for a
    superset of the first. The narrow claim is deliberate: every arm must constrain the
    same key field and one of them must be that field's recognised equality, so the group
    provably contains the key predicate rather than replacing it. A group spanning two
    fields, or naming a non-key column, is rejected exactly as before.
    """
    leaves = {
        m.group(1).lower()
        for m in re.finditer(
            r"(?:`?\w+`?\.)*`?(\w+)`?\s*" + _OPERATORS + r"\s*" + _LITERAL, group
        )
    }
    if len(leaves) != 1:
        return False
    leaf = next(iter(leaves))
    if leaf not in key_leaves:
        return False
    eq = r"(?:`?\w+`?\.)*`?" + re.escape(leaf) + r"`?\s*(?:==|=|\bIN\b)\s*" + _LITERAL
    return bool(re.search(eq, group, re.IGNORECASE))


def _collapse_key_widening_ors(text: str, key_leaves: set) -> Optional[str]:
    """``text`` with every admissible ``OR`` group replaced by the key equality it widens.

    Returns ``None`` when any ``OR`` group is not such a widening, the conservative answer,
    unchanged from when a single ``OR`` anywhere disqualified outright. Innermost groups
    only (``[^()]*``), so a group whose arms carry their own parentheses is unread and
    therefore refused, and a function call's parentheses are never touched.
    """
    inner_or = re.compile(r"\(([^()]*\bOR\b[^()]*)\)", re.IGNORECASE)
    out = str(text)
    # Bounded: a real query nests a handful of groups, not thousands.
    for _ in range(16):
        m = inner_or.search(out)
        if m is None:
            break
        if not _or_group_widens_one_key_field(m.group(1), key_leaves):
            return None
        leaf = next(
            iter(
                {
                    g.group(1).lower()
                    for g in re.finditer(
                        r"(?:`?\w+`?\.)*`?(\w+)`?\s*" + _OPERATORS + r"\s*" + _LITERAL,
                        m.group(1),
                    )
                }
            )
        )
        # Substituted rather than deleted, so the per-field check below still sees the key
        # field constrained; the group's own equality arm is what licensed it.
        start, end = m.start(), m.end()
        out = out[:start] + f"{leaf} = 'x'" + out[end:]
    # A disjunct we never resolved (top-level, or nested past the bound) still disqualifies.
    return None if re.search(r"\bOR\b", out, re.IGNORECASE) else out


def key_was_enforced(text: str, fields: List[str]) -> bool:
    """True if the final query constrains every one of ``fields`` conjunctively.

    Read only to interpret an empty result: a keyed lookup returning zero rows is a finding
    only when the key was in the predicate; otherwise zero rows are uninformative. Returns
    ``True`` only for a shape it positively recognises; an unread query returns ``False``.

    A disjunct that widens one key field's equality is admitted: entailment for an empty
    result runs through supersets, so ``sign = 'X' OR sign LIKE 'X%'`` returning nothing
    proves ``sign = 'X'`` returns nothing. A group spanning two fields or a non-key column
    is refused (see :func:`_or_group_widens_one_key_field`).
    """
    if not text or len(fields or []) < 1:
        return False
    key_leaves = {leaf_of(f).lower() for f in fields}
    # An OR still disqualifies unless every group is a widening of one key field's own
    # equality, in which case the group is collapsed back to that equality and the
    # per-field check below proceeds over a text that is provably no narrower.
    if re.search(r"\bOR\b", text, re.IGNORECASE):
        collapsed = _collapse_key_widening_ors(text, key_leaves)
        if collapsed is None:
            return False
        text = collapsed
    for field in fields:
        leaf = leaf_of(field)
        constrained = re.compile(
            r"`?\w*`?\.?`?" + re.escape(leaf) + r"`?\s*(?:==|=|\bIN\b)\s*" + _LITERAL,
            re.IGNORECASE,
        )
        if not constrained.search(text):
            return False
    return True


def enforce_conjunction(text: str, fields: List[str], source_name: str = "?") -> str:
    """Rewrite a top-level ``OR`` between two composite-key predicates into ``AND``.

    ``fields`` are the resolved backend field names of the source's
    ``require_all_entities`` (already mapped from entity types). Only an ``OR`` directly
    between predicates on the first two required fields is rewritten, and only when both
    are present; anything else is left untouched.
    """
    if not text or len(fields or []) < 2:
        return text
    leaves = [leaf_of(f) for f in fields]
    operand = r"`?\w*`?\.?`?{leaf}`?\s*(?:==|=|\bIN\b)\s*" + _LITERAL
    pattern = re.compile(
        "("
        + operand.format(leaf=re.escape(leaves[0]))
        + r")\s+OR\s+("
        + operand.format(leaf=re.escape(leaves[1]))
        + ")",
        re.IGNORECASE,
    )
    fixed, n = pattern.subn(r"\1 AND \2", text)
    if n:
        logger.warning(
            "Source '%s' is keyed by %s: rewrote %d OR -> AND in the generated query "
            "(an OR on a composite-key lookup returns other keys' rows).",
            source_name,
            " + ".join(leaves),
            n,
        )
    return fixed


# --- value combinations -----------------------------------------------------------------
#
# Input is `[[{type, value, value_form}, ...], ...]` from `follow_up.harvest_value_tuples` or
# `api_call_generator._incident_value_tuples`. Via `field_mapping.value_tuple_columns`.


def _column_literals(conjunct: str) -> Optional[tuple]:
    """``(column_text, [(raw_literal, value), ...])`` if ``conjunct`` is one flat equality.

    ``None`` for everything else. The residue decides: the text must be fully accounted for
    by one column, one ``=``/``==``/``IN``, and its literal(s). Checking the residue rather
    than an allowlist is the same choice :func:`_plain_subgroup_operands` makes: an allowlist
    fails open on the one nobody thought of.

    The raw literal is returned beside its unquoted value because the rewrite splices it back
    verbatim; re-quoting would choose a literal syntax the generator has already chosen.
    """
    text = str(conjunct or "").strip()
    while text.startswith("(") and text.endswith(")"):
        inner = text[1:-1]
        if _split_top_level_or(inner) is None:
            break
        text = inner.strip()
    match = re.match(
        r"^(?P<col>(?:`[^`]+`|\"[^\"]+\"|[\w$]+)(?:\s*\.\s*(?:`[^`]+`|\"[^\"]+\"|[\w$]+))*)"
        r"\s*(?:(?P<eq>==|=)|(?P<in>\bIN\b))\s*(?P<rhs>.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    rhs = match.group("rhs").strip()
    if match.group("in"):
        if not (rhs.startswith("(") and rhs.endswith(")")):
            return None
        items = _split_top_level(rhs[1:-1], ",")
        if items is None:
            return None
    else:
        items = [rhs]
    literals: List[tuple] = []
    for item in items:
        raw = item.strip()
        if len(raw) < 2 or raw[0] not in "'\"" or raw[-1] != raw[0]:
            return None  # a bare token, a number, a function call, another column
        value = raw[1:-1]
        if raw[0] in value:
            return None  # an escaped/doubled quote; not a literal this may re-emit
        literals.append((raw, value))
    if not literals:
        return None
    return match.group("col").strip(), literals


def _slot_columns(conjunct: str) -> Optional[List[tuple]]:
    """The ``[(column, [(raw, value), ...]), ...]`` one conjunct constrains, or ``None``.

    A slot may name more than one column: :func:`relax_form_conjunction`, which runs before
    this guard, turns an AND across one entity type's several form columns into
    ``(form_a OR form_b)``, a single conjunct. Topology is read from the query, not from any
    declaration: columns OR-ed inside a slot stay OR-ed, slots stay AND-ed, so each arm is a
    conjunction of subsets of what it replaces.

    An OR-group is all-or-nothing: one unreadable arm declines the whole conjunct, leaving it
    AND-ed exactly as generated.
    """
    text = str(conjunct or "").strip()
    while text.startswith("(") and text.endswith(")"):
        inner = text[1:-1]
        if _split_top_level_or(inner) is None:
            break  # unbalanced parentheses; not a pair this may unwrap
        text = inner.strip()
    arms = _split_top_level_or(text)
    if arms is None:
        return None
    if len(arms) == 1:
        parsed = _column_literals(text)
        return None if parsed is None else [parsed]
    out: List[tuple] = []
    for arm in arms:
        parsed = _column_literals(arm)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def _paren_and_chain(conjunct: str) -> Optional[List[str]]:
    """The conjuncts of ``conjunct`` when it is nothing but a parenthesised AND chain.

    ``None`` for everything else. A group is conjunctive with its parent only when the parent's
    operator reaches it unmodified: the text must open with the parenthesis, close with its
    match, and split at depth zero into two or more AND parts with no OR between them.
    Refused shapes: ``NOT (a AND b)`` (opens with a token), ``f(a AND b) = 1`` and
    ``(a) = (b)`` (parenthesis closes before the end), ``(a AND b) OR c`` (splits on an OR).

    The early-close test is shadowed by :func:`_split_top_level_or` but kept for clarity:
    it states the accepted shape where a reader looks for it.
    """
    text = str(conjunct or "").strip()
    if not (text.startswith("(") and text.endswith(")")):
        return None
    depth = 0
    quote = ""
    for index, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return None  # the opening parenthesis closes before the end: not one group
    if depth or quote:
        return None
    inner = text[1:-1]
    disjuncts = _split_top_level_or(inner)
    if disjuncts is None or len(disjuncts) != 1:
        return None
    parts = _split_top_level(inner, "AND")
    if parts is None or len(parts) < 2:
        return None
    return parts


def _flatten_and_conjuncts(body: str) -> Optional[List[str]]:
    """``body``'s AND chain, descending into every conjunct that is only a nested AND chain.

    ``AND`` is associative, so ``a AND (b AND c)`` constrains the same rows as ``a AND b AND c``.
    A component the generator wrote inside a parenthesised group is otherwise invisible as a
    slot of a combination. A flattened conjunct loses its parentheses on reassembly, so descent
    is refused whenever that would change the predicate (see :func:`_paren_and_chain`).
    ``None`` only when the body is unbalanced, same contract as :func:`_split_top_level`.
    """
    conjuncts = _split_top_level(body, "AND")
    if conjuncts is None:
        return None
    out: List[str] = []
    for conjunct in conjuncts:
        nested = _paren_and_chain(conjunct)
        if nested is None:
            out.append(conjunct)
            continue
        deeper = _flatten_and_conjuncts(" AND ".join(nested))
        out.extend(nested if deeper is None else deeper)
    return out


def enforce_value_tuples(
    text: str,
    tuples: Sequence[Sequence[tuple]],
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """Rewrite AND-ed per-column value lists into the combinations that actually occurred.

    ``tuples`` is ``[[(column, [literal, ...]), ...], ...]`` from
    :func:`field_mapping.value_tuple_columns`: one inner list per observed combination, one
    pair per component, the literals being that component's acceptable spellings on that column.

    ``col_a IN ('x','y') AND col_b IN ('p','q')`` becomes
    ``((col_a = 'x' AND col_b = 'p') OR (col_a = 'y' AND col_b = 'q'))`` when those are the
    two combinations observed; the other pairings belong to other parties.

    Returns ``text`` unchanged, with a log line, on every shape it cannot prove it is narrowing.
    """
    if not text or not tuples:
        return text
    harvested: Dict[str, set] = {}
    for tup in tuples:
        for column, literals in tup:
            harvested.setdefault(leaf_of(column).lower(), set()).update(
                str(v).lower() for v in literals
            )
    if len(harvested) < 2:
        return text
    bodies = _filter_bodies(text, dialect)
    if not bodies:
        # Said out loud, because a guard that declines and a guard with nothing to do produce
        # the identical query and the identical silence.
        logger.warning(
            "Source '%s': carries %d harvested value combination(s) but the generated query "
            "offers no filter body to rewrite (no WHERE, or a subquery whose predicates "
            "constrain its own result set, and choosing between them is guesswork). "
            "The AND-ed value lists are asked as generated, i.e. as their cross product.",
            source_name,
            len(tuples),
        )
        return text
    out = text
    for start, end in reversed(bodies):
        new = _rewrite_tuple_body(out[start:end], tuples, harvested, source_name)
        if new is not None:
            out = out[:start] + new + out[end:]
    if out == text:
        # The summary line, so a run cannot narrow nothing in silence: every body declined for a
        # reason of its own above, and this is the consequence they share.
        logger.warning(
            "Source '%s': carries %d harvested value combination(s) and %d filter body(ies), "
            "and narrowed none of them — the AND-ed value lists are asked as their cross "
            "product. The reason per body is logged above.",
            source_name,
            len(tuples),
            len(bodies),
        )
    return out


def _filter_bodies(text: str, dialect: str) -> List[tuple]:
    """``(start, end)`` of every filter-predicate span, in order.

    Stronger than :func:`_filter_region`: a splice needs the exact bounds of a body whose
    top-level ANDs are the query's conjunction. For SQL that is the outer WHERE body of every
    arm of a top-level set operation up to the first trailing clause; a subquery's own WHERE
    is not offered (its predicates constrain the inner result set). For pipe dialects every
    ``WHERE`` stage is its own body.

    An arm holding a subquery (more than one WHERE) declines the whole statement, because a
    set operation returns the union of its arms' rows.
    """
    if dialect == "esql":
        spans: List[tuple] = []
        depth = 0
        quote = ""
        stage_start = 0
        i = 0
        while i <= len(text):
            at_end = i == len(text)
            ch = "" if at_end else text[i]
            if quote:
                if ch == quote:
                    quote = ""
                i += 1
                continue
            if ch in "'\"":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if at_end or (ch == "|" and depth == 0):
                stage = text[stage_start:i]
                match = re.match(r"(\s*WHERE\b)", stage, re.IGNORECASE)
                if match:
                    spans.append((stage_start + match.end(), i))
                stage_start = i + 1
            i += 1
        return spans
    arms, operators = _set_operation_arms(text)
    spans: List[tuple] = []
    offset = 0
    for index, arm in enumerate(arms):
        if len(re.findall(r"\bWHERE\b", arm, re.IGNORECASE)) > 1:
            return []  # a subquery: which WHERE the components belong to is guesswork
        where_end = _outer_where_end(arm)
        if where_end is not None:
            start = offset + where_end
            spans.append((start, start + _tail_index(arm[where_end:])))
        offset += len(arm) + (len(operators[index]) if index < len(operators) else 0)
    return spans


def _rewrite_tuple_body(
    body: str,
    tuples: Sequence[Sequence[tuple]],
    harvested: Dict[str, set],
    source_name: str,
) -> Optional[str]:
    """One filter body's rewrite, or ``None`` to leave it exactly as generated.

    Every ``None`` logs a reason: a guard that declines is indistinguishable from one with
    nothing to do, and the caller adds a summary line when no body was rewritten.
    """
    conjuncts = _flatten_and_conjuncts(body)
    if conjuncts is None or len(conjuncts) < 2:
        logger.warning(
            "Source '%s': carries %d harvested value combination(s) but this filter body is "
            "%s, so there is no conjunction of value lists to narrow — the body is published "
            "as generated. The combinations reach a query only where the components are "
            "AND-ed conjuncts of one filter chain.",
            source_name,
            len(tuples),
            "not a readable AND chain" if conjuncts is None else "a single conjunct",
        )
        return None
    # Conjunct index is the slot: columns sharing one index were written in one OR-group
    # by the generator or by `relax_form_conjunction`, and the arms below reproduce that topology.
    positions: Dict[str, int] = {}
    asked: Dict[str, Dict[str, str]] = {}
    written: Dict[str, str] = {}
    for index, conjunct in enumerate(conjuncts):
        parsed = _slot_columns(conjunct)
        if parsed is None:
            continue
        leaves = [leaf_of(column).lower() for column, _literals in parsed]
        if len(set(leaves)) != len(leaves):
            continue  # one column twice inside the group; ambiguous arm assignment
        if any(leaf not in harvested or leaf in positions for leaf in leaves):
            # Not a component, a column already read, or an OR-group arm with no harvest.
            # All-or-nothing per conjunct: stays AND-ed as generated.
            continue
        for leaf, (column, literals) in zip(leaves, parsed):
            positions[leaf] = index
            asked[leaf] = {value.lower(): raw for raw, value in literals}
            written[leaf] = column
    # A combination is across slots; two components inside one OR-group share one slot index.
    slots: Dict[int, List[str]] = {}
    for leaf, index in positions.items():
        slots.setdefault(index, []).append(leaf)
    if len(positions) < 2 or len(slots) < 2:
        logger.warning(
            "Source '%s': carries %d harvested value combination(s) but this filter body "
            "constrains the component column(s) %s in %d slot(s) — a combination is a fact "
            "ACROSS slots, so there is nothing to narrow and the body is published as "
            "generated. A component the query constrains nowhere gets no predicate from this "
            "guard, and one buried inside a group it cannot read is not a conjunct.",
            source_name,
            len(tuples),
            ", ".join(sorted(positions)) or "(none)",
            len(slots),
        )
        return None
    # Combinations in one run need not share one shape, so a shape (the slots one family covers)
    # is chosen and only those slots are replaced; every other conjunct keeps the generator's
    # predicate. Choice is by family size, then slot count, then slot order.
    members: List[Dict[str, tuple]] = []
    for tup in tuples:
        by_leaf: Dict[str, tuple] = {}
        for column, literals in tup:
            leaf = leaf_of(column).lower()
            if leaf not in positions:
                continue  # this target did not constrain that component
            for value in literals:
                raw = asked[leaf].get(str(value).lower())
                if raw is not None:
                    by_leaf[leaf] = (raw, str(value).lower())
                    break
        if by_leaf:
            members.append(by_leaf)
    shapes: Dict[frozenset, int] = {}
    for by_leaf in members:
        covered = frozenset(
            index
            for index, leaves in slots.items()
            if any(leaf in by_leaf for leaf in leaves)
        )
        if len(covered) > 1:  # one slot is not a combination across slots
            shapes[covered] = shapes.get(covered, 0) + 1
    families: Dict[frozenset, List[Dict[str, tuple]]] = {}
    for shape in shapes:
        family: List[Dict[str, tuple]] = []
        for by_leaf in members:
            kept = {
                leaf: pair
                for leaf, pair in by_leaf.items()
                if positions[leaf] in shape
            }
            if all(any(leaf in kept for leaf in slots[index]) for index in shape):
                family.append(kept)
        families[shape] = family
    if not families:
        logger.warning(
            "Source '%s': the generated query constrains %s but none of the %d harvested "
            "combination(s) names two of those slots together, so it was left as written — "
            "there is no combination ACROSS slots to narrow to.",
            source_name,
            " + ".join(sorted(positions)),
            len(tuples),
        )
        return None
    chosen = max(
        families,
        key=lambda shape: (len(families[shape]), len(shape), [-i for i in sorted(shape)]),
    )
    family = families[chosen]
    for shape, count in sorted(shapes.items(), key=lambda kv: sorted(kv[0])):
        if shape == chosen:
            continue
        logger.info(
            "Source '%s': %d harvested combination(s) name a different shape (%s) and did not "
            "narrow this query — the %d combination(s) of shape (%s) did. Every slot outside "
            "that shape keeps the predicate the generator wrote.",
            source_name,
            count,
            " + ".join(sorted(leaf for index in shape for leaf in slots[index])),
            len(family),
            " + ".join(sorted(leaf for index in chosen for leaf in slots[index])),
        )
    # Proof: every literal the query put on a replaced slot must be one the chosen family carries.
    # Checked against the family's own harvest, not global: a literal only in a discarded
    # combination appears in no arm, so the global harvest would falsely pass the proof.
    chosen_leaves = [leaf for index in chosen for leaf in slots[index]]
    family_values: Dict[str, set] = {leaf: set() for leaf in chosen_leaves}
    for by_leaf in family:
        for leaf, (_raw, value) in by_leaf.items():
            family_values[leaf].add(value)
    for leaf in chosen_leaves:
        extra = set(asked[leaf]) - family_values[leaf]
        if extra:
            logger.warning(
                "Source '%s': left the generated query as written — column %s carries "
                "literal(s) (%s) that this pass did not harvest, so rewriting its value "
                "lists into the observed combinations could not be shown to narrow the "
                "query. The cross product of the AND-ed lists is asked as generated.",
                source_name,
                leaf,
                ", ".join(sorted(extra)),
            )
            return None
    arms: List[str] = []
    seen: set = set()
    for by_leaf in family:
        key = tuple(sorted((leaf, raw) for leaf, (raw, _value) in by_leaf.items()))
        if key in seen:
            continue
        seen.add(key)
        # Slots in query order; a slot's columns in arm order. Alphabetical is equally
        # deterministic but obscures which predicate became which component.
        pieces: List[str] = []
        for index in sorted(chosen):
            present = [leaf for leaf in slots[index] if leaf in by_leaf]
            inner = " OR ".join(f"{written[leaf]} = {by_leaf[leaf][0]}" for leaf in present)
            pieces.append(inner if len(present) == 1 else "(" + inner + ")")
        arms.append("(" + " AND ".join(pieces) + ")")
    if not arms:
        logger.warning(
            "Source '%s': the generated query constrains %s but NONE of the %d harvested "
            "combinations is fully within the literals it asked for, so it was left as "
            "written — every row it can return is a pairing that did not occur, and that is "
            "a finding about the pass rather than something to rewrite away.",
            source_name,
            " + ".join(sorted(positions)),
            len(tuples),
        )
        return None
    # Product across the replaced slots, sum within each: counted per slot (not per column) so
    # multi-column slots are not multiply-counted as a cross product the query never asked for.
    product = 1
    for index in chosen:
        product *= sum(len(asked[leaf]) for leaf in slots[index])
    if len(arms) >= product:
        return None  # the lists already enumerate only real combinations: nothing to narrow
    # The group takes the place of the first component conjunct; the rest are dropped.
    # Every other conjunct keeps its original position.
    replaced = set(chosen)
    first = min(replaced)
    group = arms[0] if len(arms) == 1 else "(" + " OR ".join(arms) + ")"
    rebuilt: List[str] = []
    for index, conjunct in enumerate(conjuncts):
        if index == first:
            rebuilt.append(group)
        elif index in replaced:
            continue
        elif conjunct.strip():
            rebuilt.append(conjunct.strip())
    logger.warning(
        "Source '%s': rewrote %d AND-ed value list(s) on %s into the %d combination(s) that "
        "were actually observed — the cross product asked for %d, so %d pairing(s) belonging "
        "to other parties were dropped from the query's key space.",
        source_name,
        len(chosen_leaves),
        " + ".join(sorted(chosen_leaves)),
        len(arms),
        product,
        product - len(arms),
    )
    return " " + " AND ".join(rebuilt) + " "


# --- fabricated literals ----------------------------------------------------------------
# Drops a predicate whose column is bound to type A but whose literal belongs to type B.
# Uses path (not leaf) to distinguish struct fields. Do not pass the event window as an
# incident value: removing a time bound leaves the query unscoped.


def _path_segments(field: str) -> List[str]:
    """A dotted path as normalised segments (``t.`a`.b`` -> ``['t', 'a', 'b']``)."""
    return [
        seg.strip('`"[] ').lower() for seg in str(field).split(".") if seg.strip('`"[] ')
    ]


def _contains_run(long_: List[str], short: List[str]) -> bool:
    """Is ``short`` a contiguous run of segments inside ``long_``?"""
    n = len(short)
    return n <= len(long_) and any(
        long_[i : i + n] == short for i in range(len(long_) - n + 1)
    )


def _binding_paths(bindings: Dict[str, Any]) -> List[tuple]:
    """``[(path segments, entity type), ...]`` for every field one source binds.

    Flattens the per-value-form shape (``{type: {form: [field]}}``) the same way
    ``field_priors_for`` does: which form a column holds is not this guard's question, only
    which type; a login on the sign's column is the cross-form defect ``render_filters``
    owns, and re-deciding it here with a coarser test could only undo that work.
    """
    out: List[tuple] = []
    for etype, entry in (bindings or {}).items():
        fields: List[str] = []
        if isinstance(entry, dict):
            for per_form in entry.values():
                if isinstance(per_form, str):
                    fields.append(per_form)
                elif isinstance(per_form, list):
                    fields.extend(str(f) for f in per_form)
        elif isinstance(entry, str):
            fields = [entry]
        elif isinstance(entry, list):
            fields = [str(f) for f in entry]
        for field in fields:
            segments = _path_segments(field)
            if segments and (segments, etype) not in out:
                out.append((segments, etype))
    return out


def _agrees_on_leaf(pred: List[str], bound: List[str]) -> bool:
    """Do the two paths name the same column, one being the other's tail?

    The two spellings of a declared field: the generated query may carry a table alias or
    struct prefix the pack omitted (``t.a.b`` vs ``a.b``), or write the bare leaf of a path the
    pack declared in full (``b`` vs ``a.b``). Either side may be the longer, so both directions
    are tested.
    """
    if not pred or not bound:
        return False
    return pred[-len(bound) :] == bound or bound[-len(pred) :] == pred


def _types_for_column(path: List[str], binding_paths: List[tuple]) -> List[str]:
    """Entity types this source declares for the column at ``path``.

    More than one is normal and deliberately permissive: one struct is commonly bound by two
    types at once (the party it identifies, and that party's organisational unit), and a
    literal matching either keeps its predicate.

    The specific reading wins: a binding that names this column decides. An enclosing binding
    is consulted only when none names the leaf directly, and only if the pack names the leaf
    nowhere else. A struct is bound by what it identifies, not by every field inside it: a
    struct declared as the actor still carries the actor's own attributes (values of other
    types), so descending and claiming the whole subtree would drop correct predicates.
    """
    naming: List[str] = []
    enclosing: List[str] = []
    for segments, etype in binding_paths:
        if _agrees_on_leaf(path, segments):
            if etype not in naming:
                naming.append(etype)
        elif _contains_run(path, segments):
            if etype not in enclosing:
                enclosing.append(etype)
    if naming:
        return naming
    leaf_declared = any(segments[-1] == path[-1] for segments, _ in binding_paths)
    return [] if leaf_declared else enclosing


def _types_by_value(incident_values: Dict[str, Iterable[str]]) -> Dict[str, List[str]]:
    """``{literal: [entity type, ...]}`` for the incident's own values, case-folded.

    A value carried by two entity types keeps both, and that is what makes the test safe on
    a domain whose types overlap: the predicate survives if ANY of the value's types is
    bound to the column.
    """
    out: Dict[str, List[str]] = {}
    for etype, values in (incident_values or {}).items():
        for value in values or []:
            key = str(value).strip().strip("'\"").lower()
            if not key or key == "*":
                continue
            bucket = out.setdefault(key, [])
            if etype not in bucket:
                bucket.append(etype)
    return out


def _literal_value_types(literal: str, by_value: Dict[str, List[str]]) -> List[str]:
    """Which of the incident's entity types this literal is a value of (possibly none).

    Shared by every guard that has to label a generated literal with the entity type it came
    from, so the unwrapping rule below is stated once. Two guards read it: the cross-entity
    strip and the same-column conjunction; a difference between their readings would be
    invisible: one would drop a predicate the other had just made mandatory.

    A substring predicate carries the value wrapped (``*v*`` / ``%v%``); the wrapper is not
    part of the value. Only leading and trailing wildcards are stripped, so an interior
    wildcard still fails the lookup: a pattern is the value only when the value is the whole
    of it.
    """
    raw = str(literal).strip().strip("'\"")
    types = by_value.get(raw.lower()) or []
    if not types:
        unwrapped = raw.strip("*%").strip()
        if unwrapped and unwrapped != raw:
            types = by_value.get(unwrapped.lower()) or []
    return list(types)


def _cross_entity_verdict(
    column: str,
    literal: str,
    binding_paths: List[tuple],
    by_value: Dict[str, List[str]],
) -> Optional[str]:
    """Why this predicate is a fabricated filter, or ``None`` to keep it.

    Returns the reason string used in the log line, so caller and log cannot disagree about
    which predicates were dropped and why.
    """
    path = _path_segments(column)
    if not path:
        return None
    column_types = _types_for_column(path, binding_paths)
    if not column_types:
        return None  # the source declares nothing about this column
    # The ends-only wildcard unwrap lives in `_literal_value_types`, shared with the
    # same-column conjunction guard so the two cannot read one literal differently.
    value_types = _literal_value_types(literal, by_value)
    if not value_types:
        return None  # not one of the incident's values; nothing to contradict
    if set(t.lower() for t in value_types) & set(t.lower() for t in column_types):
        return None
    return (
        f"column '{column}' holds {'/'.join(column_types)} on this source, but the literal "
        f"is the incident's {'/'.join(value_types)}"
    )


def strip_fabricated_predicates(
    text: str,
    bindings: Dict[str, Any],
    incident_values: Dict[str, Iterable[str]],
    source_name: str = "?",
    pipe_stages: bool = False,
) -> str:
    """Drop textual predicates that put one entity type's value on another's column.

    ``bindings`` is the source's ``entity_bindings`` verbatim; ``incident_values`` is
    ``{entity type: [value, ...]}`` from the incident's extracted entities; the window is
    deliberately not among them (see the section note).

    Conservative in the same spirit as the other guards: only ``... AND <col> <op> <lit>``,
    a leading ``WHERE <col> <op> <lit> AND``, and (with ``pipe_stages``) a whole ``| WHERE``
    stage that is nothing but the predicate. A predicate inside an OR-group is left alone and
    logged; an OR branch is not narrowing the result on its own, so the fabricated literal
    there costs nothing but a wasted disjunct, and rewriting someone's boolean tree to
    remove it is the more expensive mistake.
    """
    if not text or not bindings:
        return text
    binding_paths = _binding_paths(bindings)
    by_value = _types_by_value(incident_values)
    if not binding_paths or not by_value:
        return text
    out = text
    dropped: List[str] = []
    # Whole path captured (not leaf) so a field inside a bound struct is not confused with a
    # same-named leaf elsewhere. `_LITERAL` also matches bare words, so `TIMESTAMP'...'` is
    # seen as keyword + quoted literal; the cast is consumed explicitly.
    pred = (
        r"(?P<col>(?:`?\w+`?\.)*`?\w+`?)\s*"
        + _OPERATORS
        + r"\s*(?:DATE|TIMESTAMP)?\s*(?P<lit>"
        + _LITERAL
        + r")"
    )

    def _reason(m: "re.Match") -> Optional[str]:
        return _cross_entity_verdict(
            m.group("col"), m.group("lit"), binding_paths, by_value
        )

    shapes: List[str] = []
    if pipe_stages:
        shapes.append(r"\|\s*WHERE\s+" + pred + r"\s*(?=\||$)")
    shapes += [
        r"\s+AND\s+" + pred,
        r"(?<=\bWHERE\s)" + pred + r"\s+AND\s+",
    ]
    for shape in shapes:
        pattern = re.compile(shape, re.IGNORECASE)
        while True:
            match = why = None
            for candidate in pattern.finditer(out):
                reason = _reason(candidate)
                if reason:
                    match, why = candidate, reason
                    break
            if match is None:
                break
            dropped.append(why)
            out = out[: match.start()] + " " + out[match.end() :]
    for why in dropped:
        logger.warning(
            "Source '%s': dropped a FABRICATED predicate from the generated query — %s. "
            "As an equality such a filter matches nothing and reports 0 rows as a success, "
            "which reads downstream as 'the source had nothing to say'; as a substring it can "
            "instead be true of every row and delete the narrowing of the predicates beside "
            "it. Both are the same defect and neither is visible in the result.",
            source_name,
            why,
        )
    # Anything left in a shape we do not rewrite: say so, for the same reason the evidence
    # strip does. A silently over-filtered source is the failure being prevented.
    region = re.split(r"\bWHERE\b", out, maxsplit=1, flags=re.IGNORECASE)
    if len(region) > 1:
        for m in re.finditer(pred, region[1], re.IGNORECASE):
            reason = _cross_entity_verdict(
                m.group("col"), m.group("lit"), binding_paths, by_value
            )
            if reason:
                logger.warning(
                    "Source '%s': a fabricated predicate remains in a shape too complex to "
                    "rewrite safely (%s); rows the check needs may be missing.",
                    source_name,
                    reason,
                )
    return out


# --- same-column conjunction -------------------------------------------------------------
#
# When a source binds every key member to one column: rewrites an OR-group into AND between
# entity types, each type's forms OR-ed inside. Spare arms kept beside the conjunction.


_SAME_COLUMN_ARM = re.compile(
    r"^(?P<col>(?:`[^`]+`|\"[^\"]+\"|[\w$]+)(?:\s*\.\s*(?:`[^`]+`|\"[^\"]+\"|[\w$]+))*)"
    r"(?:\s*(?:==|=)\s*|\s+(?:LIKE|ILIKE|RLIKE)\s+)"
    r"(?P<lit>'(?:[^']|'')*'|\"[^\"]*\")$",
    re.IGNORECASE,
)


def _strip_wrapping_parens(text: str) -> str:
    """``text`` with any fully-enclosing parenthesis pairs removed.

    Only a pair that wraps the whole string goes: ``(a) AND (b)`` opens and closes before the
    end, so it is returned untouched. Quote-aware: a parenthesis inside a literal must not
    shift the depth.
    """
    out = str(text or "").strip()
    while out.startswith("(") and out.endswith(")"):
        depth = 0
        quote = ""
        wraps = True
        for i, ch in enumerate(out):
            if quote:
                if ch == quote:
                    quote = ""
                continue
            if ch in "'\"":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(out) - 1:
                    wraps = False
                    break
        if not wraps or depth != 0 or quote:
            break
        out = out[1:-1].strip()
    return out


def _same_column_arm(arm: str) -> Optional[tuple]:
    """``(column, literal, text)`` if ``arm`` is one flat single-literal comparison.

    ``None`` for everything else, and the caller then leaves the whole group as generated. The
    accepted operators are the ones a same-column membership test is written with
    (``=``, ``==``, ``LIKE``, ``ILIKE``, ``RLIKE``); a substring match is not optional here: on a
    blob column, which is the shape this guard exists for, it is the only form available.

    ``IN`` is absent: one ``IN`` list can hold values of two required types, and splitting it
    would re-quote literals this guard splices verbatim. Declining costs a group that could
    have been narrowed; splitting risks a predicate the backend never accepted.

    Returns the arm as written (outer parentheses stripped), because that text is spliced back.
    """
    text = _strip_wrapping_parens(arm)
    if not text:
        return None
    match = _SAME_COLUMN_ARM.match(text)
    if not match:
        return None
    return match.group("col"), match.group("lit"), text


def _same_column_conjuncts(
    arms: List[str],
    field: str,
    members: List[str],
    by_value: Dict[str, List[str]],
) -> Optional[tuple]:
    """Split ``arms`` into the key's own groups and the spare arms.

    Returns ``(groups, spare)`` or ``None`` to leave the group alone (unlabellable arm, arm on
    a different column, or fewer than two key types present). A spare arm (foreign type) is kept
    beside the conjunction: declining on it left the full OR standing. An arm carrying no
    recognised type declines the whole group.
    """
    bound = _path_segments(field)
    if not bound:
        return None
    wanted = {str(m).lower() for m in members}
    order: List[str] = []
    groups: Dict[str, List[str]] = {}
    spare: List[str] = []
    for column, literal, text in arms:
        if not _agrees_on_leaf(_path_segments(column), bound):
            return None
        all_types = _literal_value_types(literal, by_value)
        if not all_types:
            return None
        types = [t for t in all_types if str(t).lower() in wanted]
        if not types:
            spare.append(text)
            continue
        # A value carried by two types is ambiguous; sorting makes the choice deterministic
        # so the same query is rewritten the same way on every run.
        key = sorted(str(t).lower() for t in types)[0]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(text)
    if len(order) < 2:
        return None
    return [groups[k] for k in order], spare


# --- form-split relaxation ---------------------------------------------------------------
#
# One type spread across several columns: an AND requires one row to carry both spellings.
# Widened because `key_was_enforced` entailment runs through supersets.


# A negated group's body must not be rewritten: `NOT (a AND b)` splits at depth zero into one
# conjunct, declining the whole text; the walker then offers the body, and any merge there
# lands under the negation. `_rewrite_and_regions` skips any group the text negates.
_NEGATED_GROUP = re.compile(r"\bNOT\s*$", re.IGNORECASE)
# A UNION is a wall for a rewrite that moves text (see `relax_form_conjunction`); guards that
# only edit in place skip this, except `enforce_default_filters`, which splices onto every arm.
_UNION_TOKEN = re.compile(r"\bUNION\b", re.IGNORECASE)
_WHERE_TOKEN = re.compile(r"\bWHERE\b", re.IGNORECASE)


def _where_prefix(part: str) -> tuple:
    """``(prefix, arm)``: the clause text before this conjunct, and the conjunct itself.

    The first conjunct of an AND chain carries ``SELECT ... FROM t WHERE``, so an anchored arm
    reader sees no arm and the guard silently declines when the form predicate was written
    first. The prefix is split off and returned for the caller to re-attach. The cut is the
    last ``WHERE`` at paren depth zero (a subquery in the projection may carry its own).
    No ``WHERE`` means no prefix, which is every conjunct but the first.
    """
    text = str(part or "")
    depth = 0
    quote = ""
    cut = -1
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"`":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "wW":
            match = _WHERE_TOKEN.match(text, i)
            if match and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
                cut = match.end()
    if cut < 0:
        return "", text
    # The whitespace after the keyword belongs to the prefix, or re-attaching it spells `WHERE(`.
    while cut < len(text) and text[cut].isspace():
        cut += 1
    return text[:cut], text[cut:]


def _column_form(column: str, forms: List[tuple]) -> Optional[str]:
    """Which declared value form this column belongs to, or ``None``.

    ``forms`` is ``[(form name, [bound column, ...])]`` for one entity type on one source.
    Comparison via :func:`_agrees_on_leaf` so a generated path prefix does not prevent a match.
    A column claimed by two forms yields ``None`` (pack defect; merging on an ambiguous binding
    could OR two arms that identify genuinely different subjects).
    """
    pred = _path_segments(column)
    if not pred:
        return None
    hits = [
        name
        for name, columns in forms
        for bound in columns or []
        if _agrees_on_leaf(pred, _path_segments(bound))
    ]
    unique = []
    for name in hits:
        if name not in unique:
            unique.append(name)
    return unique[0] if len(unique) == 1 else None


def _form_conjunct_groups(
    parts: List[str],
    entity_type: str,
    forms: List[tuple],
    by_value: Dict[str, List[str]],
) -> List[List[int]]:
    """Indices of the conjuncts to merge, one list per mergeable group.

    A conjunct qualifies on four facts: (1) parses as a flat positive comparison after
    :func:`_where_prefix` strips any ``SELECT ... WHERE`` prefix; (2) its column is bound to
    exactly one declared form of ``entity_type`` (:func:`_column_form`); (3) its literal is
    one of the incident's own values for that type; (4) the group spans at least two distinct
    forms (several values of one form on one column are already an IN list).

    An unparseable conjunct is not a decline: the first conjunct of any real query carries the
    full ``SELECT ... FROM ... WHERE`` prefix and parses as nothing; unlabelled conjuncts are
    left verbatim.
    """
    wanted = str(entity_type).lower()
    by_form: Dict[str, List[int]] = {}
    order: List[str] = []
    for index, part in enumerate(parts):
        arm = _same_column_arm(_where_prefix(part)[1])
        if arm is None:
            continue
        column, literal, _text = arm
        form = _column_form(column, forms)
        if not form:
            continue
        types = [t.lower() for t in _literal_value_types(literal, by_value)]
        if wanted not in types:
            continue
        if form not in by_form:
            by_form[form] = []
            order.append(form)
        by_form[form].append(index)
    if len(order) < 2:
        return []
    return [by_form[name] for name in order]


def relax_form_conjunction(
    text: str,
    form_bindings: List[tuple],
    incident_values: Dict[str, Iterable[str]],
    source_name: str = "?",
) -> str:
    """Rewrite an AND between two value forms of one entity type into an OR.

    ``form_bindings`` is ``[(entity type, [(form, [column, ...]), ...])]`` from
    ``field_mapping.form_split_bindings``. Reaches every AND chain: the whole text, then each
    parenthesised group outermost first; a rewritten region is not re-entered.

    The only guard here that moves text. A merge is refused when the parts it spans contain the
    ``UNION`` token: merging across a branch boundary hoists an arm into another branch. This
    over-refuses one harmless shape (``UNION`` inside a subquery). Safe direction: a refused
    merge leaves the defect logged; a wrong splice produces a silently wrong result set.
    """
    if not text or not form_bindings or not incident_values:
        return text
    by_value = _types_by_value(incident_values)
    if not by_value:
        return text
    applied: List[str] = []
    declined: List[str] = []

    def rewrite(body: str) -> str:
        # All split types in the region, not just the first: relaxing one while leaving another
        # AND-ed fixes only half the defect. Each pass re-splits because a merge changes the
        # conjunct indices the next type would read.
        for _ in range(len(form_bindings)):
            spans = _split_top_level_spans(body, "AND")
            if spans is None or len(spans) < 2:
                return body
            parts = [body[start:end] for start, end in spans]
            merged_this_pass = False
            for entity_type, forms in form_bindings:
                groups = _form_conjunct_groups(parts, entity_type, forms, by_value)
                if not groups:
                    continue
                merged = sorted(i for group in groups for i in group)
                # Check the parts the merge would span, not the whole body: a query may
                # legitimately UNION two branches and hold a mergeable pair inside one of them.
                lo, hi = merged[0], merged[-1] + 1
                spanned = "".join(parts[lo:hi])
                if _UNION_TOKEN.search(spanned):
                    note = (
                        f"{entity_type}: the conjuncts span a UNION, so merging them would move "
                        f"text between branches"
                    )
                    if note not in declined:
                        declined.append(note)
                    continue
                split = [_where_prefix(parts[i]) for i in merged]
                # Only the first conjunct of a branch can carry a clause prefix; a prefix on any
                # merged part after the first means a branch boundary is inside the span. A prefix
                # silently dropped here would delete a whole ``SELECT ... WHERE``.
                if any(prefix for prefix, _arm in split[1:]):
                    declined.append(
                        f"{entity_type}: a merged conjunct carries its own clause prefix, so the "
                        f"merge would cross a statement boundary"
                    )
                    continue
                group = (
                    split[0][0]
                    + "("
                    + " OR ".join(_strip_wrapping_parens(arm).strip() for _p, arm in split)
                    + ")"
                )
                kept = [
                    group if i == merged[0] else parts[i]
                    for i in range(len(parts))
                    if i == merged[0] or i not in merged
                ]
                applied.append(
                    f"{entity_type} ({len(merged)} conjuncts across "
                    f"{len(groups)} of its value forms)"
                )
                body = " AND ".join(p.strip() for p in kept if p.strip())
                merged_this_pass = True
                break
            if not merged_this_pass:
                return body
        return body

    out = _rewrite_and_regions(text, rewrite)
    for note in applied:
        logger.warning(
            "Source '%s': %s were AND-ed across their own columns — two forms of one entity "
            "type are the SAME subject, so an AND requires one row to carry both spellings and "
            "deletes every row carrying only one. Rewritten to an OR inside the key.",
            source_name,
            note,
        )
    for note in declined:
        logger.info(
            "Source '%s': left a cross-form AND as generated — %s", source_name, note
        )
    return out


def _rewrite_and_regions(text: str, rewrite) -> str:
    """Offer every AND chain in ``text`` to ``rewrite``: the whole text, then each paren group.

    AND-side counterpart of :func:`_rewrite_or_groups`. The outermost region is the whole string,
    not a parenthesised group, because a top-level AND chain needs no parentheses.
    ``rewrite(body)`` returns a replacement body or ``body`` unchanged. A region inside one
    already rewritten is not re-entered. Offsets are resolved against the original text and
    spliced highest-first so no earlier splice shifts a later one.
    """
    regions = [(0, len(text), False)] + [(s, e, True) for s, e in _paren_spans(text)]
    accepted: List[tuple] = []
    for start, end, parenthesised in regions:
        if any(s <= start and end <= e for s, e, _ in accepted):
            continue
        # A negated group's body is not rewritable from here: `NOT (a AND b)` holds one conjunct
        # at depth zero, so the whole text declines; when the walker then offers the body, any OR
        # the caller writes lands under the negation, inverting it.
        if parenthesised and _NEGATED_GROUP.search(text[:start]):
            continue
        whole = text[start:end]
        body = whole[1:-1] if parenthesised else whole
        new = rewrite(body)
        if new == body:
            continue
        accepted.append((start, end, f"({new})" if parenthesised else new))
    out = text
    for start, end, new in sorted(accepted, key=lambda a: -a[0]):
        out = out[:start] + new + out[end:]
    return out


def relax_form_conjunction_dsl(
    dsl: Any,
    form_bindings: List[tuple],
    incident_values: Dict[str, Iterable[str]],
    source_name: str = "?",
) -> Any:
    """:func:`relax_form_conjunction` for a Query DSL body; the same rule on the other route.

    ``must`` is the AND, so the rewrite moves the cross-form clauses out of ``must`` into a nested
    ``should`` with ``minimum_should_match: 1``, the same shape
    :func:`enforce_conjunction_same_column_dsl` produces in reverse. ``filter`` is left alone: it
    holds mandatory window and slice predicates, not identity arms. ``must_not`` is left alone:
    collecting an exclusion into the merge target would make excluded values mandatory, collapsing
    "exclude both" into "match either".

    Returns a new dict; the input is not mutated. No ``UNION`` refusal applies on this route.
    """
    if not isinstance(dsl, dict) or not form_bindings or not incident_values:
        return dsl
    by_value = _types_by_value(incident_values)
    if not by_value:
        return dsl
    applied: List[str] = []

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "bool" not in node or not isinstance(node.get("bool"), dict):
            return {k: walk(v) for k, v in node.items()}
        inner = dict(node["bool"])
        must = inner.get("must")
        must = list(must) if isinstance(must, list) else ([must] if must else [])
        pairs = [_dsl_clause_pairs(c) for c in must]
        # All split types; no clause merged twice. A value carried by two types can be labelled
        # by both, and moving one clause into two groups would duplicate it.
        taken: List[int] = []
        groups: List[tuple] = []
        for entity_type, forms in form_bindings:
            wanted = str(entity_type).lower()
            by_form: Dict[str, List[int]] = {}
            order: List[str] = []
            for index, clause_pairs in enumerate(pairs):
                if index in taken:
                    continue
                # One clause, possibly several (field, literal) pairs; a `terms` list yields one
                # per value. Labelled by the first pair matching this type on a form column,
                # because the clause moves whole.
                form = next(
                    (
                        f
                        for field, literal in clause_pairs
                        for f in [_column_form(field, forms)]
                        if f
                        and wanted
                        in [t.lower() for t in _literal_value_types(literal, by_value)]
                    ),
                    None,
                )
                if not form:
                    continue
                if form not in by_form:
                    by_form[form] = []
                    order.append(form)
                by_form[form].append(index)
            if len(order) < 2:
                continue
            merged = sorted(i for name in order for i in by_form[name])
            taken.extend(merged)
            groups.append((entity_type, merged, len(order)))
        for entity_type, merged, forms_n in groups:
            applied.append(f"{entity_type} ({len(merged)} clauses across {forms_n} forms)")
        if groups:
            kept = [c for i, c in enumerate(must) if i not in taken]
            promoted = [
                {
                    "bool": {
                        "should": [must[i] for i in merged],
                        # Explicit: the default is 1 only while the bool carries no `must`; a
                        # nested group may gain one. Left implicit, both forms would have to
                        # co-occur again.
                        "minimum_should_match": 1,
                    }
                }
                for _t, merged, _n in groups
            ]
            if kept or len(promoted) > 1:
                inner["must"] = kept + promoted
            else:
                # One group and nothing else mandatory: collapse `must` to avoid a one-element
                # list holding a `should`.
                inner.pop("must", None)
                inner["should"] = promoted[0]["bool"]["should"]
                inner["minimum_should_match"] = 1
        # Recurse over every occurrence list, merged or not. A nested `bool` inside `filter` or
        # `should` can carry the same defect; returning early on a merge would leave it unvisited.
        out = {k: walk(v) for k, v in node.items() if k != "bool"}
        out["bool"] = {k: walk(v) for k, v in inner.items()}
        return out

    result = walk(dsl)
    for note in applied:
        logger.warning(
            "Source '%s': %s were in `must` together — two forms of one entity type are the "
            "SAME subject, so requiring both deletes every document carrying one. Moved into a "
            "`should` with minimum_should_match 1.",
            source_name,
            note,
        )
    return result


def enforce_conjunction_same_column(
    text: str,
    groups: List[tuple],
    incident_values: Dict[str, Iterable[str]],
    source_name: str = "?",
) -> str:
    """Rewrite an OR between two required entity types on one column into an AND.

    ``groups`` is :func:`same_column_conjunctions`' output: ``[(field, [entity type, ...])]``
    for columns carrying two or more key members. Reach is parenthesised OR-groups
    (:func:`_rewrite_or_groups`), same as :func:`enforce_identity_scope`: a top-level ``OR``
    beside a window bound already means something else (``A OR B AND C`` = ``A OR (B AND C)``).
    """
    if not text or not groups or not incident_values:
        return text
    by_value = _types_by_value(incident_values)
    if not by_value:
        return text
    applied: List[str] = []
    declined: List[str] = []

    def rewrite(body: str, whole: str) -> str:
        parts = _split_top_level_or(body)
        if parts is None or len(parts) < 2:
            return whole
        arms = [_same_column_arm(p) for p in parts]
        if any(a is None for a in arms):
            declined.append("an arm is not a flat single-literal comparison")
            return whole
        for field, members in groups:
            split = _same_column_conjuncts(arms, field, members, by_value)
            if split is None:
                continue
            conjuncts, spare = split
            rendered = [
                c[0] if len(c) == 1 else "(" + " OR ".join(c) + ")" for c in conjuncts
            ]
            conjunction = "(" + " AND ".join(rendered) + ")"
            note = f"{field} ({len(conjuncts)} of {'/'.join(members)})"
            if not spare:
                applied.append(note)
                return conjunction
            # The key's members AND each other; an arm the key does not name stays an alternative
            # to the whole conjunction, not a conjunct of it. Pulled inside the AND it would
            # require the subject and that value to co-occur in one record.
            applied.append(f"{note}, {len(spare)} arm(s) left OR-ed beside it")
            return "(" + " OR ".join([conjunction] + spare) + ")"
        declined.append("no declared same-column key accounts for two or more arms")
        return whole

    out = _rewrite_or_groups(text, rewrite)
    for note in applied:
        logger.warning(
            "Source '%s': rewrote an OR into an AND on a SHARED column — %s. Every member of "
            "this source's declared key binds to that one column, so the tuple resolved to a "
            "single field and the ordinary conjunction guard had nothing to rewrite; the "
            "OR-ed form returns every record naming ANY member, which is a full result of "
            "other parties' rows rather than a visible failure.",
            source_name,
            note,
        )
    if declined and not applied:
        # A decline is indistinguishable from nothing-to-do without this log line.
        logger.info(
            "Source '%s': left %d OR-group(s) as generated on a shared key column — %s.",
            source_name,
            len(declined),
            "; ".join(sorted(set(declined))),
        )
    return out


# --- a predicate that constrains nothing -------------------------------------------------
#
# Drops a vacuous arm (e.g. `col IS NOT NULL`) only when a real predicate on the same column
# survives. A vacuous predicate standing alone is left (dropping it leaves the column free).

# A comparison that is true of every row where the column is populated. Anchored at both ends
# by the caller, which supplies the column path.
_VACUOUS_RHS = (
    r"(?:"
    r"IS\s+NOT\s+NULL"  # populated at all
    r"|(?:<>|!=)\s*(?:''|\"\")"  # not the empty string
    r"|(?:NOT\s+)?LIKE\s*'%'"  # matches any non-null value
    r"|(?:NOT\s+)?LIKE\s*\"%\""
    r")"
)


def _vacuous_arms(body: str) -> List[tuple]:
    """``[(leaf, arm text), ...]`` for every top-level arm of ``body`` that constrains nothing.

    An arm qualifies only when it is nothing but a vacuous comparison on one column, so an arm
    that also says something real (``actor IS NOT NULL AND unit = 'X'``) is not included.
    """
    parts = _split_top_level_or(body)
    if parts is None:
        return []
    pattern = re.compile(
        r"^\s*\(*\s*(?P<col>(?:`?\w+`?\.)*`?(?P<leaf>\w+)`?)\s*"
        + _VACUOUS_RHS
        + r"\s*\)*\s*$",
        re.IGNORECASE,
    )
    out: List[tuple] = []
    for part in parts:
        m = pattern.match(part)
        if m:
            out.append((m.group("leaf").lower(), part.strip()))
    return out


def strip_vacuous_disjuncts(text: str, source_name: str = "?") -> str:
    """Drop an OR-ed arm that is true of every populated row, when a real arm survives.

    An arm is dropped only from a group where another arm constrains the same column with a
    real comparison, so the group narrows and the column stays bounded; every other shape is
    left as generated.
    """
    if not text or not re.search(r"\bOR\b", text, re.IGNORECASE):
        return text
    group_re = re.compile(
        r"\(\s*((?:[^()]|\([^()]*\))*?\bOR\b(?:[^()]|\([^()]*\))*?)\s*\)", re.DOTALL
    )
    dropped: List[str] = []

    def _rewrite(match: "re.Match") -> str:
        body = match.group(1)
        vacuous = _vacuous_arms(body)
        if not vacuous:
            return match.group(0)
        parts = _split_top_level_or(body)
        if parts is None:
            return match.group(0)
        vacuous_text = {arm for _, arm in vacuous}
        kept = [p for p in parts if p.strip() not in vacuous_text]
        if not kept:
            # Every arm was vacuous; dropping them all would leave the column unconstrained.
            return match.group(0)
        # A real comparison on the same column must survive, otherwise the drop changes
        # which column is bounded rather than only how tightly.
        kept_text = " OR ".join(kept)
        for leaf, arm in vacuous:
            constrained = re.compile(
                r"(?<!\w)(?:`?\w+`?\.)*`?" + re.escape(leaf) + r"`?\s*" + _ANY_COMPARISON,
                re.IGNORECASE,
            )
            if not constrained.search(kept_text):
                return match.group(0)
            dropped.append(arm)
        rebuilt = kept_text.strip()
        return "(" + rebuilt + ")" if len(kept) > 1 else rebuilt

    out = group_re.sub(_rewrite, text)
    for arm in dropped:
        logger.warning(
            "Source '%s': dropped the OR-ed arm `%s` — it is true of every row where that "
            "column is populated, so it SUBSUMES the arm naming the subject and the query "
            "asks for the whole table. That does not return zero rows, it returns a full "
            "result which reads downstream as an answer about the subject.",
            source_name,
            " ".join(arm.split()),
        )
    return out


# --- partition pruning ------------------------------------------------------------------
#
# Specs: ``{"name": ..., "role": "date|year|month|day", "type": "DATE", "pad_days": N}``.
# ``pad_days_after`` overrides the upper pad for columns updated after the event date.

# Clause keywords that end a WHERE clause; the constant finds where a predicate may be appended.
_SQL_TAIL_KEYWORDS = (
    "GROUP BY",
    "HAVING",
    "QUALIFY",
    "WINDOW",
    "ORDER BY",
    "LIMIT",
    "OFFSET",
    "UNION",
    "INTERSECT",
    "EXCEPT",
)
# Any comparison operator; wider than _OPERATORS because partition bounds use >= / <= / BETWEEN.
_ANY_COMPARISON = (
    r"(?:>=|<=|<>|<|>|==|=|!=|\bIN\b|\bBETWEEN\b|\bLIKE\b|\bRLIKE\b|\bIS\b)"
)


def _as_date(value: Any) -> Optional[date]:
    """Parse ``YYYY-MM-DD`` (or an ISO timestamp's date part). None if unusable."""
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


# Column types whose stored value is a number: literals are rendered bare. One list so
# `_quote` and `partition_bounds` agree; an unquoted date is parsed as arithmetic.
_NUMERIC_COL_PREFIXES = (
    "INT",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "LONG",
    "SHORT",
    "DECIMAL",
    "NUMERIC",
)


def _is_numeric_col(col_type: str) -> bool:
    return (col_type or "").upper().startswith(_NUMERIC_COL_PREFIXES)


def _quote(value: str, col_type: str, dialect: str) -> str:
    """Render a scalar literal for one dialect + column type."""
    t = (col_type or "").upper()
    if _is_numeric_col(t):
        return value
    if dialect == "esql":
        # `ES|QL` string literals are double-quoted and it has no DATE'...' form.
        return f'"{value}"'
    if t.startswith("DATE"):
        return f"DATE'{value}'"
    if t.startswith("TIMESTAMP"):
        return f"TIMESTAMP'{value}'"
    return f"'{value}'"


def _calendar_parts(role: str, start: date, end: date, col_type: str) -> List[str]:
    """Distinct values of one calendar part across ``[start, end]``.

    Zero-padded to the width the ``year=``/``month=``/``day=`` convention uses, unless the
    column is numeric (then the bare number is the stored value).
    """
    numeric = _is_numeric_col(col_type)
    seen: List[str] = []
    cursor = start
    while cursor <= end:
        if role == "year":
            raw, width = cursor.year, 4
        elif role == "month":
            raw, width = cursor.month, 2
        else:
            raw, width = cursor.day, 2
        text = str(raw) if numeric else str(raw).zfill(width)
        if text not in seen:
            seen.append(text)
        cursor += timedelta(days=1)
    return seen


def partition_bounds(
    partitions: Sequence[Dict[str, Any]], date_from: Any, date_to: Any
) -> List[Dict[str, Any]]:
    """Resolve each partition spec against the incident window, dialect-independently.

    Returns ``[{"name", "type", "kind", ...}]`` where ``kind`` is either:

    * ``"range"`` with ``low``/``high`` ISO date strings, or
    * ``"in"`` with ``values``: the distinct calendar parts the window spans.

    This is the single place the window maths lives, so every dialect formatter (SQL text,
    ``ES|QL``, Query DSL, an encoded REST query) bounds the same partitions the same way. Specs
    with no name, or a window that cannot be parsed at all, are skipped.
    """
    start, end = _as_date(date_from), _as_date(date_to)
    if start is None and end is None:
        return []
    start = start or end
    end = end or start
    if end < start:
        start, end = end, start
    out: List[Dict[str, Any]] = []
    for spec in partitions or []:
        name = str((spec or {}).get("name") or "").strip()
        if not name:
            continue
        role = str(spec.get("role") or "date").strip().lower()
        col_type = str(spec.get("type") or "")
        try:
            pad = int(spec.get("pad_days", 1) or 0)
        except (TypeError, ValueError):
            pad = 1
        # The upper pad may be declared separately for a column recording when the row was
        # written: later rows are the subject's current state, so cutting them off truncates
        # the record. Absent, the symmetric pad applies.
        pad_after = spec.get("pad_days_after")
        try:
            pad_after = pad if pad_after is None else int(pad_after or 0)
        except (TypeError, ValueError):
            pad_after = pad
        lo, hi = (
            start - timedelta(days=max(pad, 0)),
            end + timedelta(days=max(pad_after, 0)),
        )
        if role in ("year", "month", "day"):
            values = _calendar_parts(role, lo, hi, col_type)
            if not values:
                continue
            out.append({"name": name, "type": col_type, "kind": "in", "values": values})
        elif _is_numeric_col(col_type):
            # A date literal on a numeric column is arithmetic, producing a contradictory predicate
            # that deletes every row. Skipped rather than bounded: an unbounded partition costs
            # extra storage scanned; a contradictory one costs the entire result.
            logger.warning(
                "Partition column '%s' is %s, so the incident's date window cannot be a "
                "literal on it — it is left UNBOUNDED rather than bounded by a predicate "
                "that matches nothing. If it stores a calendar part declare "
                "role: year|month|day; if it stores an instant declare it under "
                "epoch_time_columns; if it is neither (a version or shard counter) this is "
                "correct and costs only extra storage scanned.",
                name,
                col_type or "numeric",
            )
            continue
        else:
            out.append(
                {
                    "name": name,
                    "type": col_type or "DATE",
                    "kind": "range",
                    "low": lo.isoformat(),
                    "high": hi.isoformat(),
                }
            )
    return out


def partition_predicates(
    partitions: Sequence[Dict[str, Any]],
    date_from: Any,
    date_to: Any,
    dialect: str = "sql",
) -> List[Dict[str, str]]:
    """Render :func:`partition_bounds` as textual predicates for a SQL-like dialect.

    Returns ``[{"name": <column>, "predicate": <text>}, ...]``. Pure: no query text is
    touched here, so the same predicates can be injected into a query or shown to the
    generator.
    """
    out: List[Dict[str, str]] = []
    for bound in partition_bounds(partitions, date_from, date_to):
        name, col_type = bound["name"], bound["type"]
        if bound["kind"] == "in":
            literals = [_quote(v, col_type, dialect) for v in bound["values"]]
            predicate = (
                f"{name} = {literals[0]}"
                if len(literals) == 1
                else f"{name} IN ({', '.join(literals)})"
            )
        else:
            predicate = (
                f"{name} >= {_quote(bound['low'], col_type, dialect)} AND "
                f"{name} <= {_quote(bound['high'], col_type, dialect)}"
            )
        out.append({"name": name, "predicate": predicate})
    return out


def _filter_region(text: str, dialect: str) -> str:
    """The part of a query where filtering happens (everything after the first WHERE).

    For a pipe dialect every ``| WHERE`` stage counts. Used only to test whether a column
    is already constrained, so a loose answer is fine as long as it errs toward "yes".
    """
    if dialect == "esql":
        stages = re.split(r"\|", text)
        return " ".join(s for s in stages if re.match(r"\s*WHERE\b", s, re.IGNORECASE))
    parts = re.split(r"\bWHERE\b", text, maxsplit=1, flags=re.IGNORECASE)
    return parts[1] if len(parts) > 1 else ""


def _is_constrained(text: str, field: str, dialect: str) -> bool:
    """True if ``field`` already appears in a comparison inside the filter region.

    The leading ``(?<!\\w)`` is required: without it the column name matches as a suffix of a
    longer identifier and the function returns true, skipping the partition bound. Only the
    left edge needs guarding; the right edge is already pinned by the comparison operator.
    """
    region = _filter_region(text, dialect)
    if not region:
        return False
    pattern = (
        r"(?<!\w)(?:`?\w+`?\.)*`?"
        + re.escape(leaf_of(field))
        + r"`?\s*"
        + _ANY_COMPARISON
    )
    return bool(re.search(pattern, region, re.IGNORECASE))


_SUBQUERY_TOKEN = "__subquery__"


def _mask_subqueries(text: str) -> str:
    """Replace every balanced ``(...SELECT...)`` group with an opaque token.

    A subquery carries its own ``WHERE``, and :func:`_filter_region` splits on the first one,
    so inner predicates stay in the region and a comparison inside a subquery would read as
    constraining the outer result. The token carries no parenthesis, so depth counting
    downstream is unaffected.
    """
    out = text
    while True:
        replaced = False
        for start in [i for i, c in enumerate(out) if c == "("]:
            depth = 0
            for i in range(start, len(out)):
                depth += (out[i] == "(") - (out[i] == ")")
                if depth == 0:
                    inner = out[start : i + 1]
                    if re.search(r"\bSELECT\b", inner, re.IGNORECASE):
                        out = out[:start] + _SUBQUERY_TOKEN + out[i + 1 :]
                        replaced = True
                    break
            if replaced:
                break
        if not replaced:
            return out


def _is_constrained_conjunctively(text: str, fields: Iterable[str], dialect: str) -> bool:
    """Does every row this query can return satisfy a comparison on one of ``fields``?

    Stronger than :func:`_is_constrained`, which asks only whether the column appears anywhere
    in the filter region. Here the filter region is split on AND at depth zero and each conjunct
    is tested independently: a conjunct qualifies when it is one comparison, or a parenthesised
    group whose every top-level OR arm constrains one of the fields (a synonym family: whichever
    arm matched, the row is the subject's). Anything else reads as not constraining, which is
    the safe direction: an anchor AND-ed onto a query already anchored only narrows.

    A set operation is evaluated per arm, with every arm required: a UNION returns the union of
    its arms' rows, so one arm that does not bind the identity returns rows without it.
    """
    columns = [f for f in fields if f]
    if not columns:
        return False
    masked = _mask_subqueries(text)
    arms, _operators = _set_operation_arms(masked) if dialect == "sql" else ([masked], [])
    if len(arms) > 1:
        return all(_arm_is_constrained_conjunctively(a, columns, dialect) for a in arms)
    return _arm_is_constrained_conjunctively(masked, columns, dialect)


def _arm_is_constrained_conjunctively(
    text: str, columns: List[str], dialect: str
) -> bool:
    """:func:`_is_constrained_conjunctively` for one arm."""
    region = _filter_region(text, dialect)
    if not region:
        return False

    def _binds(fragment: str) -> bool:
        stripped = fragment.strip()
        while stripped.startswith("(") and stripped.endswith(")"):
            inner = stripped[1:-1]
            if _split_top_level_or(inner) is None:
                break
            stripped = inner.strip()
        arms = _split_top_level_or(stripped)
        if arms is None:
            return False
        if len(arms) > 1:
            # A group binds the identity only if every arm does.
            return all(_binds(arm) for arm in arms)
        conjuncts = _split_top_level(stripped, "AND")
        if conjuncts is None:
            return False
        if len(conjuncts) > 1:
            return any(_binds(c) for c in conjuncts)
        return any(_is_constrained(f"WHERE {stripped}", col, "sql") for col in columns)

    conjuncts = _split_top_level(region, "AND")
    if conjuncts is None:
        return False
    return any(_binds(c) for c in conjuncts)


def _outer_where_end(text: str) -> Optional[int]:
    """Index just past the OUTER statement's ``WHERE``, or ``None`` if it has none.

    Depth-aware, so a subquery's own ``WHERE`` is skipped rather than mistaken for the outer
    one. The other guards in this module decline outright when a statement holds more than one
    ``WHERE``, which is correct for them because for a bound either one is a defensible place to inject
    and choosing is guesswork. It is not guesswork: exactly one ``WHERE`` sits at parenthesis
    depth zero, and that is the one whose predicate decides which rows come back.
    """
    depth = 0
    quote = ""
    i = 0
    n = len(text)
    upper = text.upper()
    while i < n:
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and upper.startswith("WHERE", i):
            before_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            after = i + 5
            after_ok = after >= n or not (text[after].isalnum() or text[after] == "_")
            if before_ok and after_ok:
                return after
        i += 1
    return None


def _tail_index(text: str) -> int:
    """Index of the first top-level trailing clause (GROUP BY / ORDER BY / LIMIT / ...).

    Parenthesis depth is tracked so a keyword inside a subquery or function call is not
    mistaken for the end of the outer clause. Quoted text is skipped: a tail keyword inside a
    string literal would be read as the end of the filter region, splitting the literal in
    half. Returns ``len(text)`` when there is no trailing clause.
    """
    depth = 0
    quote = ""
    i = 0
    upper = text.upper()
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            for kw in _SQL_TAIL_KEYWORDS:
                if upper.startswith(kw, i):
                    before_ok = i == 0 or not (
                        text[i - 1].isalnum() or text[i - 1] == "_"
                    )
                    after = i + len(kw)
                    after_ok = after >= len(text) or not (
                        text[after].isalnum() or text[after] == "_"
                    )
                    if before_ok and after_ok:
                        return i
        i += 1
    return len(text)


def enforce_partition_bounds(
    text: str,
    partitions: Sequence[Dict[str, Any]],
    date_from: Any,
    date_to: Any,
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """AND a bound on every declared partition column the generated query left unbounded.

    An unbounded partition column does not change which rows match, only how much storage is
    read to find them, so the injection is semantics-preserving. Conservative: a column already
    compared in the filter region is left alone; a SQL statement with more than one ``WHERE`` is
    left alone and logged; the existing WHERE body is parenthesised before the bound is AND-ed
    on so a top-level OR inside it cannot be silently re-associated.
    """
    if not text or not partitions:
        return text
    predicates = [
        p
        for p in partition_predicates(partitions, date_from, date_to, dialect=dialect)
        if not _is_constrained(text, p["name"], dialect)
    ]
    if not predicates:
        return text
    clause = " AND ".join(p["predicate"] for p in predicates)
    names = ", ".join(p["name"] for p in predicates)

    if dialect == "esql":
        # A pipeline stage is composable by construction: insert it right after the source
        # stage so the filter runs before anything downstream.
        head, sep, rest = text.partition("|")
        out = (
            f"{head.rstrip()} | WHERE {clause} {sep}{rest}"
            if sep
            else f"{text.rstrip()} | WHERE {clause}"
        )
    else:
        if len(re.findall(r"\bWHERE\b", text, re.IGNORECASE)) > 1:
            logger.warning(
                "Source '%s': partition column(s) %s are unbounded but the query has more "
                "than one WHERE clause, so no bound was injected — the scan is not pruned "
                "and the query may be slow enough to time out.",
                source_name,
                names,
            )
            return text
        match = re.search(r"\bWHERE\b", text, re.IGNORECASE)
        if match:
            head, rest = text[: match.end()], text[match.end() :]
            cut = _tail_index(rest)
            body, tail = rest[:cut].strip(), rest[cut:]
            out = f"{head} {clause} AND ({body})"
            if tail.strip():
                out = f"{out} {tail.strip()}"
        else:
            cut = _tail_index(text)
            body, tail = text[:cut].rstrip(), text[cut:]
            out = f"{body} WHERE {clause}"
            if tail.strip():
                out = f"{out} {tail.strip()}"
    logger.warning(
        "Source '%s': injected a partition bound on %s — the generated query left it "
        "unbounded, which reads the whole table instead of the window's partitions.",
        source_name,
        names,
    )
    return out


def event_time_predicate(
    column: str, col_type: str, date_from: Any, date_to: Any, dialect: str = "sql"
) -> Optional[str]:
    """A two-sided bound on ``column`` for the incident's own window, or ``None``.

    The window is unpadded: the padded partition window is already applied by
    :func:`enforce_partition_bounds`, and the point of this guard is to stop that pad
    becoming the effective event window. Upper-exclusive against the day after ``date_to``
    so an instant column covers the whole of the last day (``<= date_to`` on a timestamp
    cuts at midnight).
    """
    start, end = _as_date(date_from), _as_date(date_to)
    if start is None and end is None:
        return None
    start = start or end
    end = end or start
    if end < start:
        start, end = end, start
    low = _quote(start.isoformat(), col_type, dialect)
    high = _quote((end + timedelta(days=1)).isoformat(), col_type, dialect)
    return f"{column} >= {low} AND {column} < {high}"


def enforce_event_time_window(
    text: str,
    column: Optional[str],
    col_type: str,
    date_from: Any,
    date_to: Any,
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """AND the incident's window onto a finer event-time column the query left unbounded.

    :func:`enforce_partition_bounds` adds a bound on the partition column with its pad;
    :func:`enforce_epoch_window` only repairs epoch-integer columns. This guard handles the
    third case: a source binding a coarse partition column and the instant the event carries
    had nothing bounding the second, so the partition's pad became the effective event window.

    Whether a column qualifies is decided upstream by ``field_mapping.event_time_column``, the
    single seam both query families resolve through; this function receives a column to bound,
    or ``None``. Conservative: a column already compared is left as generated; the existing
    WHERE body is parenthesised before the AND so a top-level OR cannot be re-associated.
    """
    if not text or not column:
        return text
    if _is_constrained(text, column, dialect):
        return text
    clause = event_time_predicate(column, col_type, date_from, date_to, dialect=dialect)
    if not clause:
        return text

    if dialect == "esql":
        head, sep, rest = text.partition("|")
        out = (
            f"{head.rstrip()} | WHERE {clause} {sep}{rest}"
            if sep
            else f"{text.rstrip()} | WHERE {clause}"
        )
    else:
        # Depth-aware: exactly one WHERE sits at depth zero and that is the one deciding
        # which rows come back, so a subquery is no reason to leave the outer window unbounded.
        end = _outer_where_end(text)
        if end is not None:
            head, rest = text[:end], text[end:]
            cut = _tail_index(rest)
            body, tail = rest[:cut].strip(), rest[cut:]
            out = f"{head} {clause} AND ({body})"
            if tail.strip():
                out = f"{out} {tail.strip()}"
        else:
            cut = _tail_index(text)
            body, tail = text[:cut].rstrip(), text[cut:]
            out = f"{body} WHERE {clause}"
            if tail.strip():
                out = f"{out} {tail.strip()}"
    logger.warning(
        "Source '%s': injected an event-time bound on %s — the generated query left it "
        "unbounded, so the partition column's pad was the effective event window and "
        "off-window rows compete for the row cap with the subject's own.",
        source_name,
        column,
    )
    return out


def enforce_event_time_window_dsl(
    dsl: Dict,
    column: Optional[str],
    col_type: str,
    date_from: Any,
    date_to: Any,
    source_name: str = "?",
) -> Dict:
    """AND a finer event-time bound into a Query DSL query. DSL twin of the above.

    Same shape as :func:`enforce_partition_bounds_dsl`: no query text to re-associate, so the
    original is nested as one clause of a new ``bool.filter`` beside the bound, preserving its
    own boolean structure exactly. Returns a new dict.
    """
    if not isinstance(dsl, dict) or not column:
        return dsl
    if f'"{column}"' in json.dumps(dsl):
        return dsl
    start, end = _as_date(date_from), _as_date(date_to)
    if start is None and end is None:
        return dsl
    start = start or end
    end = end or start
    if end < start:
        start, end = end, start
    logger.warning(
        "Source '%s': injected an event-time bound on %s into the generated query DSL — it "
        "was left unbounded, so the partition pad was the effective event window.",
        source_name,
        column,
    )
    clause = {
        "range": {
            column: {
                "gte": start.isoformat(),
                "lt": (end + timedelta(days=1)).isoformat(),
            }
        }
    }
    return {"bool": {"filter": [clause, copy.deepcopy(dsl)]}}


def partition_prompt_line(
    partitions: Sequence[Dict[str, Any]],
    date_from: Any = None,
    date_to: Any = None,
    dialect: str = "sql",
) -> str:
    """One prompt sentence naming the partition columns to bound, or ``''``.

    The injection above is the guarantee, but telling the generator produces a better query
    than a bolted-on predicate can (it can, for instance, bound the column more tightly than
    the padded window). Derived from the discovered/declared specs so no source has to
    hand-write the same prose into its ``query_hints``, where it would drift out of date with
    the physical layout; a live-discovered fact must not go stale.
    """
    if not partitions:
        return ""
    predicates = partition_predicates(partitions, date_from, date_to, dialect=dialect)
    names = ", ".join(
        str((p or {}).get("name")) for p in partitions if (p or {}).get("name")
    )
    if not names:
        return ""
    line = (
        f"MANDATORY PARTITION PRUNE — this table is PARTITIONED on: {names}. Every query "
        "MUST bound EVERY one of those columns in its WHERE clause. Partition columns are "
        "how the storage engine skips data it does not need to read: leaving one unbounded "
        "reads the ENTIRE table (terabytes) and the query times out, which returns NO rows "
        "and is indistinguishable downstream from the source having no data. Bound them IN "
        "ADDITION to the columns the request talks about — a partition column is usually NOT "
        "the same column as the event/creation time the question is about, and bounding only "
        "the latter prunes nothing."
    )
    if predicates:
        line += (
            " Suitable predicates for this incident's window: "
            + " AND ".join(p["predicate"] for p in predicates)
            + "."
        )
    return line


def merge_partition_specs(
    discovered: Sequence[Dict[str, Any]], declared: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Combine live-discovered partition columns with the pack's declarations.

    Discovery is authoritative about which columns partition a table and is preferred,
    because it cannot go stale. The pack's declaration is how a source covers what discovery
    cannot see (a VIEW does not expose its underlying table's layout) and how it supplies
    what metadata does not carry (``role``, ``pad_days``). So: union by column name, with
    the declaration's keys overriding the discovered ones for the same column.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for spec in list(discovered or []) + list(declared or []):
        name = str((spec or {}).get("name") or "").strip()
        if not name:
            continue
        merged.setdefault(name, {})
        merged[name].update({k: v for k, v in spec.items() if v is not None})
    return list(merged.values())


# --- epoch time windows -----------------------------------------------------------------
#
# Pack declares epoch columns and their unit: ``{"name": "timestamp", "unit": "milliseconds"}``.
# Bounds are computed from the incident's window, not from the prompt.

# Accepted spellings of an epoch unit -> how many of them make one second.
_EPOCH_UNITS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "epoch_second": 1,
    "ms": 1000,
    "milli": 1000,
    "millis": 1000,
    "millisecond": 1000,
    "milliseconds": 1000,
    "epoch_millis": 1000,
    "us": 1_000_000,
    "micro": 1_000_000,
    "micros": 1_000_000,
    "microsecond": 1_000_000,
    "microseconds": 1_000_000,
    "ns": 1_000_000_000,
    "nano": 1_000_000_000,
    "nanos": 1_000_000_000,
    "nanosecond": 1_000_000_000,
    "nanoseconds": 1_000_000_000,
}
# Comparison operators an epoch bound is written with (no IN/LIKE: an epoch instant is not
# enumerated, and a rewrite of a set literal would be guesswork).
_EPOCH_OP = r"(>=|<=|==|!=|=|>|<)"


def epoch_scale(unit: Any) -> Optional[int]:
    """Units per second for a declared epoch unit, or None if it is not recognised."""
    return _EPOCH_UNITS.get(str(unit or "milliseconds").strip().lower())


def _midnight_epoch(day: date, scale: int) -> int:
    """Start of ``day`` in UTC, in epoch units."""
    return (
        int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
        * scale
    )


def epoch_window(date_from: Any, date_to: Any, unit: Any = "milliseconds"):
    """The incident window as a closed epoch interval ``(low, high)``, or None.

    ``low`` is midnight UTC on ``date_from``; ``high`` is the last instant of ``date_to``
    (23:59:59 plus the sub-second remainder the unit can express), covering both end days
    completely. A single-sided window is treated as one day.
    """
    start, end = _as_date(date_from), _as_date(date_to)
    if start is None and end is None:
        return None
    start = start or end
    end = end or start
    if end < start:
        start, end = end, start
    scale = epoch_scale(unit)
    if scale is None:
        return None
    low = _midnight_epoch(start, scale)
    high = _midnight_epoch(end, scale) + 86400 * scale - 1
    return low, high


def _as_epoch_int(value: Any) -> Optional[int]:
    """An integer epoch literal, or None if the value is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value or "").strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def _date_as_epoch(value: Any, scale: int, upper: bool) -> Optional[int]:
    """An ISO date/timestamp literal converted to epoch units, or None.

    ``upper`` picks the last instant of the day rather than the first, so a
    ``<= '2026-07-27'`` bound keeps including the whole of the 27th.
    """
    day = _as_date(value)
    if day is None:
        return None
    base = _midnight_epoch(day, scale)
    return base + 86400 * scale - 1 if upper else base


def epoch_specs(
    time_columns: Sequence[Dict[str, Any]], date_from: Any, date_to: Any
) -> List[Dict[str, Any]]:
    """Resolve declared epoch columns against the window: ``[{name, unit, scale, low, high}]``.

    Specs with no name, an unknown unit, or an unparseable window are skipped; a guard that
    cannot compute the right answer must not rewrite anything.
    """
    out: List[Dict[str, Any]] = []
    for spec in time_columns or []:
        name = str((spec or {}).get("name") or "").strip()
        if not name:
            continue
        unit = (spec or {}).get("unit") or "milliseconds"
        window = epoch_window(date_from, date_to, unit)
        scale = epoch_scale(unit)
        if window is None or scale is None:
            continue
        out.append(
            {
                "name": name,
                "unit": str(unit),
                "scale": scale,
                "low": window[0],
                "high": window[1],
            }
        )
    return out


def enforce_epoch_window(
    text: str,
    time_columns: Sequence[Dict[str, Any]],
    date_from: Any,
    date_to: Any,
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """Repair epoch-column bounds in a generated query that cannot match the incident window.

    Rewrites two shapes only: a DATE/ISO literal on an epoch column (cannot compare against an
    integer), and an integer interval that does not intersect the incident's window. An interval
    that overlaps the window is left as generated, even when narrower than the full window.
    """
    if not text or not time_columns:
        return text
    out = text
    repaired: List[str] = []
    for spec in epoch_specs(time_columns, date_from, date_to):
        low, high, scale = spec["low"], spec["high"], spec["scale"]
        col = r"((?:`?\w+`?\.)*`?" + re.escape(leaf_of(spec["name"])) + r"`?)"

        # 1. Date/ISO literals: always wrong on an epoch column, so always converted.
        def to_epoch(m, low=low, high=high, scale=scale, spec=spec):
            column, op, quote, literal = m.group(1), m.group(2), m.group(3), m.group(4)
            upper = op in ("<=", "<")
            value = _date_as_epoch(literal, scale, upper)
            if value is None:
                return m.group(0)
            repaired.append(f"{spec['name']} {op} {quote}{literal}{quote}")
            return f"{column} {op} {value}"

        out = re.sub(
            col + r"\s*" + _EPOCH_OP + r"\s*(['\"])([^'\"]+)\3",
            to_epoch,
            out,
            flags=re.IGNORECASE,
        )

        # 2. Integer bounds: rewritten only when the interval misses the window entirely.
        between_re = re.compile(
            col + r"\s+BETWEEN\s+(-?\d+)\s+AND\s+(-?\d+)", re.IGNORECASE
        )
        cmp_re = re.compile(col + r"\s*" + _EPOCH_OP + r"\s*(-?\d+)", re.IGNORECASE)
        lo_eff = hi_eff = None
        for m in between_re.finditer(out):
            a, b = int(m.group(2)), int(m.group(3))
            lo_eff = a if lo_eff is None else max(lo_eff, a)
            hi_eff = b if hi_eff is None else min(hi_eff, b)
        for m in cmp_re.finditer(out):
            op, n = m.group(2), int(m.group(3))
            if op in (">=", ">"):
                lo_eff = n if lo_eff is None else max(lo_eff, n)
            elif op in ("<=", "<"):
                hi_eff = n if hi_eff is None else min(hi_eff, n)
            elif op in ("=", "=="):
                lo_eff = hi_eff = n
        if lo_eff is None and hi_eff is None:
            continue
        if (lo_eff if lo_eff is not None else low) <= high and (
            hi_eff if hi_eff is not None else high
        ) >= low:
            continue  # overlaps the incident window; the generator's bound stands

        def fix_between(m, low=low, high=high, spec=spec):
            repaired.append(f"{spec['name']} BETWEEN {m.group(2)} AND {m.group(3)}")
            return f"{m.group(1)} BETWEEN {low} AND {high}"

        def fix_cmp(m, low=low, high=high, spec=spec):
            column, op, literal = m.group(1), m.group(2), m.group(3)
            repaired.append(f"{spec['name']} {op} {literal}")
            if op in (">=", ">"):
                return f"{column} >= {low}"
            if op in ("<=", "<"):
                return f"{column} <= {high}"
            # An equality pinned to the wrong instant becomes the window, parenthesised so
            # an enclosing OR cannot be silently re-associated by the extra AND.
            return f"({column} >= {low} AND {column} <= {high})"

        out = between_re.sub(fix_between, out)
        out = cmp_re.sub(fix_cmp, out)
    if repaired:
        logger.warning(
            "Source '%s': rewrote %d epoch time bound(s) that could not match the "
            "incident's window (%s) — the generated query asked about a different span of "
            "time, which returns zero rows and reads downstream as an empty source.",
            source_name,
            len(repaired),
            "; ".join(repaired),
        )
    return out


def epoch_prompt_line(
    time_columns: Sequence[Dict[str, Any]], date_from: Any = None, date_to: Any = None
) -> str:
    """One prompt sentence giving this incident's window as epoch literals, or ``''``.

    The rewrite above is the guarantee, but handing the generator the already-converted
    numbers avoids a worked example in ``query_hints`` that would go stale.
    """
    specs = epoch_specs(time_columns, date_from, date_to)
    if not specs:
        return ""
    parts = [
        f"{s['name']} (epoch {s['unit']}): >= {s['low']} AND <= {s['high']}"
        for s in specs
    ]
    return (
        "EPOCH TIME COLUMNS — these columns store the time as an INTEGER, not a date or an "
        "ISO string, so a range on one MUST use the integer values below and nothing else. "
        "Do NOT convert a date yourself and do NOT copy an epoch value from any example: "
        "these are computed for THIS incident's window. " + "; ".join(parts) + "."
    )


# --- a literal in the wrong precision ----------------------------------------------------
#
# The long form of an identifier matches nothing on a column storing only its stem. Adds the
# stem beside the value; confined to `field_mapping.stem_literals`; refuses negated shapes.


def _quoted(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _stem_column_pattern(column: str) -> str:
    """Left-hand side of a comparison on ``column``, however the generator spelled it.

    Uses the same leaf match and ``(?<!\\w)`` guard as :func:`_is_constrained`: without it the
    name matches as a suffix of a longer identifier and the widening lands on a different column.
    """
    return r"(?<!\w)(?:`?\w+`?\.)*`?" + re.escape(leaf_of(column)) + r"`?"


def widen_stem_literals(
    text: str,
    stem_values: Dict[str, Dict[str, str]],
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """Offer the declared stem of a literal beside it, on the same column.

    ``stem_values`` is ``{real column: {long value: stem}}``, from ``field_mapping.stem_literals``.
    Two shapes only: ``col = 'v'`` becomes ``col IN ('v', '<stem>')``, and an ``IN`` list gains
    the stem. Negations declined; any other shape left as generated. Returns ``text`` unchanged
    when there is nothing to widen.
    """
    if not text or not stem_values:
        return text
    out = text
    widened: List[str] = []
    declined: List[str] = []
    for column, pairs in stem_values.items():
        usable = {
            str(long): str(stem)
            for long, stem in (pairs or {}).items()
            if long and stem and str(long) != str(stem)
        }
        if not usable:
            continue
        col_re = _stem_column_pattern(column)
        # List first: appending inside the parens cannot change the shape of anything else, and
        # `render_filters` emits `IN` for every multi-value column. `NOT` is matched on both
        # sides of the column so a negation is declined explicitly rather than silently missed.
        list_re = re.compile(
            r"(?P<not>\bNOT\s+)?"
            + col_re
            + r"\s*(?P<not2>\bNOT\s+)?\bIN\s*\((?P<body>[^()]*)\)",
            re.IGNORECASE,
        )

        def _list(m: "re.Match", column=column, usable=usable) -> str:
            if m.group("not") or m.group("not2"):
                declined.append(
                    f"{column}: a NOT IN list — widening it would narrow the result"
                )
                return m.group(0)
            body = m.group("body")
            additions = [
                stem
                for long, stem in usable.items()
                if _quoted(long) in body and _quoted(stem) not in body
            ]
            if not additions:
                return m.group(0)
            widened.append(f"{column} IN (…) += {', '.join(_quoted(a) for a in additions)}")
            insert = ", ".join(_quoted(a) for a in additions)
            return m.group(0)[: m.end("body") - m.start()] + f", {insert}" + ")"

        out = list_re.sub(_list, out)

        # Then the equality, which has to become a list. `==` is accepted because one dialect
        # here spells equality that way; nothing else is, so `!=`/`<>`/`LIKE` fall through
        # untouched by construction rather than by an exclusion list that could go stale.
        eq_re = re.compile(
            r"(?P<not>\bNOT\s+)?" + col_re + r"\s*(?P<op>==|=)\s*(?P<lit>'[^']*')",
            re.IGNORECASE,
        )

        def _eq(m: "re.Match", column=column, usable=usable) -> str:
            literal = m.group("lit")[1:-1]
            stem = usable.get(literal)
            if not stem:
                return m.group(0)
            if m.group("not"):
                declined.append(
                    f"{column}: a negated equality — widening it would narrow the result"
                )
                return m.group(0)
            if _quoted(stem) in out:
                # Already offered somewhere in this query (the generator followed the hint, or
                # a list rewrite above did it). Rewriting the equality too would publish the
                # same fact twice, which an operator reads as two predicates.
                return m.group(0)
            widened.append(f"{column} = {m.group('lit')} -> IN (…, {_quoted(stem)})")
            head = m.group(0)[: m.start("op") - m.start()].rstrip()
            return f"{head} IN ({m.group('lit')}, {_quoted(stem)})"

        out = eq_re.sub(_eq, out)

    for note in widened:
        logger.warning(
            "Source '%s': widened a predicate to its declared identity stem — %s. The value "
            "carries an optional trailing part this target may not store, and a predicate on "
            "the long form alone would match no row of such a column while reporting 0 rows "
            "as a success.",
            source_name,
            note,
        )
    for note in declined:
        logger.info(
            "Source '%s': left a predicate as generated rather than widening it to a stem — "
            "%s.",
            source_name,
            note,
        )
    return out


def widen_stem_literals_dsl(
    dsl: Any, stem_values: Dict[str, Dict[str, str]], source_name: str = "?"
) -> Any:
    """The DSL twin of :func:`widen_stem_literals`, for the same reason every twin here exists.

    A guarantee honoured on one query family and not the other is not a guarantee: on a Query
    DSL source the pack's declaration was a silent no-op with no failure to see.

    ``term`` becomes ``terms`` (the JSON equivalent of ``=`` becoming ``IN``) and a ``terms``
    list gains the stem. Anything under ``must_not`` is skipped for the reason the textual
    guard skips a negation, and the input is not mutated.
    """
    if not isinstance(dsl, (dict, list)) or not stem_values:
        return dsl
    by_leaf = {
        leaf_of(col): {
            str(long): str(stem)
            for long, stem in (pairs or {}).items()
            if long and stem and str(long) != str(stem)
        }
        for col, pairs in stem_values.items()
    }
    by_leaf = {leaf: pairs for leaf, pairs in by_leaf.items() if pairs}
    if not by_leaf:
        return dsl
    widened: List[str] = []

    def _pairs_for(field: Any) -> Dict[str, str]:
        return by_leaf.get(leaf_of(str(field)), {})

    def _widen_terms(value: Dict) -> Dict:
        out: Dict[str, Any] = {}
        for field, literals in value.items():
            pairs = _pairs_for(field)
            if not pairs or not isinstance(literals, list):
                out[field] = literals
                continue
            widened_list = list(literals)
            for long, stem in pairs.items():
                if long in widened_list and stem not in widened_list:
                    widened_list.append(stem)
                    widened.append(f"{field}: terms += {stem}")
            out[field] = widened_list
        return out

    def _promote_term(value: Dict) -> tuple:
        """``({field: [literal, stem]}, {field: literal})``: promoted, and left alone."""
        promoted: Dict[str, Any] = {}
        kept: Dict[str, Any] = {}
        for field, literal in value.items():
            stem = (
                _pairs_for(field).get(str(literal))
                if isinstance(literal, (str, int))
                else None
            )
            if stem:
                promoted[field] = [literal, stem]
                widened.append(f"{field}: term -> terms += {stem}")
            else:
                kept[field] = literal
        return promoted, kept

    def _merge_terms(out: Dict, addition: Dict) -> None:
        merged = dict(out.get("terms") or {})
        merged.update(addition)
        out["terms"] = merged

    def _walk(node: Any, negated: bool) -> Any:
        if isinstance(node, list):
            return [_walk(item, negated) for item in node]
        if not isinstance(node, dict):
            return node
        out: Dict[str, Any] = {}
        for key, value in node.items():
            # `must_not` negates everything under it, at any depth: widening there narrows
            # the result, which is the one direction this guard must never take.
            neg = negated or str(key) == "must_not"
            if not neg and str(key) == "terms" and isinstance(value, dict):
                _merge_terms(out, _widen_terms(value))
                continue
            if not neg and str(key) == "term" and isinstance(value, dict):
                promoted, kept = _promote_term(value)
                if promoted:
                    # A promoted clause changes its KEY (`term` -> `terms`), so the two are
                    # merged rather than patched, and any field with no stem stays a `term`.
                    _merge_terms(out, promoted)
                    if kept:
                        out["term"] = kept
                    continue
            out[key] = _walk(value, neg)
        return out

    result = _walk(dsl, False)
    for note in widened:
        logger.warning(
            "Source '%s': widened a DSL clause to its declared identity stem — %s. See the "
            "textual guard: a predicate on the long form alone matches no row of a column "
            "storing the stem, and reports 0 rows as a success.",
            source_name,
            note,
        )
    return result


# --- Elasticsearch Query DSL (nested JSON) ----------------------------------------------

# Leaf query clauses that constrain a field, i.e. the ones that can filter evidence out.
_DSL_FIELD_CLAUSES = (
    "term",
    "terms",
    "match",
    "match_phrase",
    "match_phrase_prefix",
    "range",
    "prefix",
    "wildcard",
    "regexp",
    "fuzzy",
    "exists",
    "terms_set",
)
# bool occurrence types whose clause lists we may prune.
_DSL_BOOL_KEYS = ("must", "filter", "should", "must_not")


def _dsl_clause_fields(clause: Any) -> List[str]:
    """Field names a single DSL leaf clause constrains (empty if it is not a leaf)."""
    if not isinstance(clause, dict):
        return []
    out: List[str] = []
    for key, body in clause.items():
        if key not in _DSL_FIELD_CLAUSES:
            continue
        if key == "exists" and isinstance(body, dict):
            field = body.get("field")
            if field:
                out.append(str(field))
        elif isinstance(body, dict):
            out.extend(str(f) for f in body)
    return out


def strip_evidence_filters_dsl(
    dsl: Dict, never_filter: Iterable[str], source_name: str = "?"
) -> Dict:
    """Remove DSL leaf clauses that filter on a ``never_filter`` field.

    Walks the query tree and prunes matching clauses out of every ``bool`` occurrence list,
    including ``should``. An emptied occurrence key is deleted rather than left as ``[]``,
    which is not a neutral filter in all ES versions. Removing a ``should`` branch narrows:
    with no ``must`` beside it the default ``minimum_should_match`` is 1, so each branch
    admits rows on its own, and narrowing is the safe direction here for the same reason as
    the textual guard. A clause in an unmodelled position (e.g. directly under the root) is
    replaced with ``match_all`` so the query stays valid. Returns a new dict; the input is
    not mutated.
    """
    fields = [f for f in (never_filter or []) if f]
    if not fields or not isinstance(dsl, dict):
        return dsl
    leaves = {leaf_of(f).lower() for f in fields}
    removed: List[str] = []

    def targets(clause: Any) -> bool:
        return any(leaf_of(f).lower() in leaves for f in _dsl_clause_fields(clause))

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict):
                    new_bool: Dict[str, Any] = {}
                    for bkey, bvalue in value.items():
                        if bkey in _DSL_BOOL_KEYS:
                            clauses = bvalue if isinstance(bvalue, list) else [bvalue]
                            kept = []
                            for c in clauses:
                                if targets(c):
                                    removed.extend(_dsl_clause_fields(c))
                                    continue
                                kept.append(walk(c))
                            # An emptied occurrence list is dropped entirely: `[]` under
                            # `must`/`should` is not a neutral filter in every ES version.
                            if kept:
                                new_bool[bkey] = kept
                        else:
                            new_bool[bkey] = walk(bvalue)
                    # An empty bool matches nothing; degrade to match_all.
                    if not any(k in new_bool for k in _DSL_BOOL_KEYS):
                        new_bool.pop("minimum_should_match", None)
                        if not new_bool:
                            out["match_all"] = {}
                            continue
                    out[key] = new_bool
                elif targets({key: value}):
                    # A forbidden leaf clause outside any bool: keep the query valid.
                    removed.extend(_dsl_clause_fields({key: value}))
                    out["match_all"] = {}
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    if removed:
        logger.warning(
            "Source '%s': stripped %d filter clause(s) on evidence field(s) %s from the "
            "generated query DSL — those fields must be RETURNED, not filtered (a filter "
            "on one deletes the rows that answer the question).",
            source_name,
            len(removed),
            ", ".join(sorted(set(removed))),
        )
    return result


def _dsl_clause_pairs(clause: Any) -> List[tuple]:
    """``[(field, literal), ...]`` a single DSL leaf clause compares, for the value test.

    Narrower than :func:`_dsl_clause_fields`, which answers "does this clause touch the
    field" and needs no literal. ``range`` and ``exists`` are absent: a range is a bound, not
    an assertion that the column equals one of the incident's values, and dropping one could
    remove the only bound on a partition column. A multi-value ``terms`` yields one pair per
    value, so a list mixing a real value with a fabricated one is judged on the fabricated one.
    """
    if not isinstance(clause, dict):
        return []
    out: List[tuple] = []
    for key, body in clause.items():
        if key not in _DSL_FIELD_CLAUSES or key in ("range", "exists"):
            continue
        if not isinstance(body, dict):
            continue
        for field, spec in body.items():
            if isinstance(spec, dict):
                # The long form: {"wildcard": {"f": {"value": "...", "boost": 1}}}
                for vkey in ("value", "query", "term", "prefix"):
                    if vkey in spec:
                        out.append((str(field), str(spec[vkey])))
                        break
            elif isinstance(spec, (list, tuple)):
                out.extend((str(field), str(v)) for v in spec)
            elif spec is not None and not isinstance(spec, bool):
                out.append((str(field), str(spec)))
    return out


def strip_fabricated_filters_dsl(
    dsl: Dict,
    bindings: Dict[str, Any],
    incident_values: Dict[str, Iterable[str]],
    source_name: str = "?",
) -> Dict:
    """Drop DSL leaf clauses that put one entity type's value on another's column.

    DSL counterpart of :func:`strip_fabricated_predicates`. A clause is pruned from its
    ``bool`` occurrence list; an emptied list is deleted; a top-level fabricated clause
    degrades to ``match_all``. Removal is safe at every position: a fabricated equality matches
    nothing, and a substring predicate on the wrong column may match everything.
    """
    if not isinstance(dsl, dict) or not bindings:
        return dsl
    binding_paths = _binding_paths(bindings)
    by_value = _types_by_value(incident_values)
    if not binding_paths or not by_value:
        return dsl
    reasons: List[str] = []

    def targets(clause: Any) -> Optional[str]:
        for field, literal in _dsl_clause_pairs(clause):
            reason = _cross_entity_verdict(field, literal, binding_paths, by_value)
            if reason:
                return reason
        return None

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict):
                    new_bool: Dict[str, Any] = {}
                    for bkey, bvalue in value.items():
                        if bkey in _DSL_BOOL_KEYS:
                            clauses = bvalue if isinstance(bvalue, list) else [bvalue]
                            kept = []
                            for c in clauses:
                                reason = targets(c)
                                if reason:
                                    reasons.append(reason)
                                    continue
                                kept.append(walk(c))
                            if kept:
                                new_bool[bkey] = kept
                        else:
                            new_bool[bkey] = walk(bvalue)
                    if not any(k in new_bool for k in _DSL_BOOL_KEYS):
                        new_bool.pop("minimum_should_match", None)
                        if not new_bool:
                            out["match_all"] = {}
                            continue
                    out[key] = new_bool
                else:
                    reason = targets({key: value})
                    if reason:
                        reasons.append(reason)
                        out["match_all"] = {}
                    else:
                        out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    for reason in reasons:
        logger.warning(
            "Source '%s': dropped a FABRICATED filter clause from the generated query DSL "
            "— %s. As an equality such a clause matches nothing and reports 0 rows as a "
            "success, which reads downstream as 'the source had nothing to say'; as a "
            "substring it can instead be true of every row and delete the narrowing of the "
            "clauses beside it. Both are the same defect and neither is visible in the result.",
            source_name,
            reason,
        )
    return result


def _dsl_vacuous_field(clause: Any) -> Optional[str]:
    """The field a clause constrains vacuously, or ``None`` if it constrains something.

    The DSL spellings of "true wherever this field is populated": an ``exists``, and a
    ``wildcard``/``prefix``/``match`` whose pattern is nothing but ``*`` or empty. A clause
    naming more than one field is not read.
    """
    if not isinstance(clause, dict) or len(clause) != 1:
        return None
    for key, body in clause.items():
        if key == "exists" and isinstance(body, dict):
            field = body.get("field")
            return str(field) if field else None
        if key not in ("wildcard", "prefix", "match", "match_phrase", "regexp"):
            return None
        pairs = _dsl_clause_pairs(clause)
        if len(pairs) != 1:
            return None
        field, literal = pairs[0]
        pattern = str(literal).strip()
        if key == "regexp":
            return field if pattern in ("", ".*", ".+", "(?s).*") else None
        if pattern in ("", "*") or (key == "wildcard" and set(pattern) == {"*"}):
            return field
    return None


def drop_orphan_minimum_should_match(dsl: Any, source_name: str = "?") -> Any:
    """Remove ``minimum_should_match`` from every ``bool`` that has no ``should`` left.

    Earlier guards remove ``should`` arms and delete emptied occurrence keys but cannot see the
    sibling ``minimum_should_match``, so a ``bool`` keeping its ``filter`` keeps the orphan.
    ``minimum_should_match: 1`` with no ``should`` requires one match from an empty set, which
    Elasticsearch resolves as zero documents. This guard runs last; each stripper need not
    manage the key itself. Returns a new object; the input is not mutated.
    """
    if not isinstance(dsl, (dict, list)):
        return dsl
    dropped: List[str] = []

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key != "bool" or not isinstance(value, dict):
                out[key] = walk(value)
                continue
            body = {k: walk(v) for k, v in value.items()}
            should = body.get("should")
            # An absent key and an empty list are the same: nothing to match. A single clause
            # written directly (not in a list) is a populated `should`.
            empty = "should" not in body or (
                isinstance(should, list) and not should
            )
            if empty:
                body.pop("should", None)
                if body.pop("minimum_should_match", None) is not None:
                    dropped.append(",".join(sorted(body)) or "empty bool")
            out[key] = body
        return out

    result = walk(dsl)
    if dropped:
        logger.warning(
            "Source %r: dropped an orphaned minimum_should_match from %d bool clause(s) whose "
            "`should` list an earlier guard emptied (remaining keys: %s). Left in place it "
            "requires at least one match from an EMPTY set of clauses, which on Elasticsearch "
            "matches NO document — measured 108 rows -> 0 on one live index — so the query "
            "would have reported zero rows as a successful, empty answer.",
            source_name,
            len(dropped),
            "; ".join(dropped[:5]),
        )
    return result


def strip_vacuous_should_clauses(dsl: Dict, source_name: str = "?") -> Dict:
    """:func:`strip_vacuous_disjuncts` for a Query DSL body; same claim, same conservatism.

    Only a ``should`` list is read, because only a ``should`` is an OR. An ``exists`` under
    ``must``/``filter`` is an AND-ed populated-ness requirement, which narrows and is often
    correct. The same two conditions as the textual side apply: a real clause on the same field
    must survive in the list, and a list that is nothing but vacuous clauses is left alone.
    """
    if not isinstance(dsl, dict):
        return dsl
    dropped: List[str] = []

    def _prune(clauses: List[Any]) -> List[Any]:
        vacuous = [(i, _dsl_vacuous_field(c)) for i, c in enumerate(clauses)]
        vacuous = [(i, f) for i, f in vacuous if f]
        if not vacuous or len(vacuous) == len(clauses):
            return clauses
        out = list(clauses)
        for index, field in sorted(vacuous, reverse=True):
            leaf = leaf_of(field).lower()
            survivors = [
                c
                for j, c in enumerate(out)
                if j != index and _dsl_vacuous_field(c) is None
            ]
            if not any(
                leaf_of(f).lower() == leaf
                for c in survivors
                for f in _dsl_clause_fields(c)
            ):
                continue
            dropped.append(field)
            out.pop(index)
        return out

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict):
                    new_bool: Dict[str, Any] = {}
                    for bkey, bvalue in value.items():
                        if bkey == "should" and isinstance(bvalue, list):
                            new_bool[bkey] = [walk(c) for c in _prune(bvalue)]
                        else:
                            new_bool[bkey] = walk(bvalue)
                    out[key] = new_bool
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    for field in dropped:
        logger.warning(
            "Source '%s': dropped a vacuous `should` clause on '%s' from the generated "
            "query DSL — it matches every document where that field is populated, so it "
            "SUBSUMES the clause naming the subject and the query asks for the whole index. "
            "That returns a full result which reads downstream as an answer about the "
            "subject.",
            source_name,
            field,
        )
    return result


def key_was_enforced_dsl(dsl: Any, fields: List[str]) -> bool:
    """:func:`key_was_enforced` for a Query DSL body; same claim, same conservatism.

    ``must``/``filter`` (and the top-level ``query``) are conjunctive; ``must_not`` is an
    exclusion, not a lookup. Exception: a ``should`` whose every clause constrains the same key
    field counts as conjunctive (a widening of that field). Entailment runs through supersets:
    a superset returning nothing proves the subset returns nothing. Exception applies only in a
    conjunctive position; nested under another ``should`` it is part of a genuine disjunction.
    """
    if not isinstance(dsl, dict) or len(fields or []) < 1:
        return False
    wanted = {leaf_of(f).lower() for f in fields}
    seen: set = set()
    disqualified = [False]

    def _widens_one_key_field(clauses: Any) -> Optional[str]:
        """The single key field a ``should`` list widens, or ``None``.

        Mirrors :func:`_or_group_widens_one_key_field`: every arm must be a leaf clause on
        one and the same key field. A ``bool`` arm is not a leaf, so a nested tree is not
        read and therefore not admitted.
        """
        items = clauses if isinstance(clauses, list) else [clauses]
        leaves: set = set()
        for c in items:
            if not isinstance(c, dict) or any(k == "bool" for k in c):
                return None
            got = {leaf_of(f).lower() for f in _dsl_clause_fields(c)}
            if len(got) != 1:
                return None
            leaves |= got
        if len(leaves) != 1:
            return None
        leaf = next(iter(leaves))
        return leaf if leaf in wanted else None

    def walk(node: Any, conjunctive: bool) -> None:
        if disqualified[0]:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict):
                    should = value.get("should")
                    if should not in (None, [], {}):
                        widened = _widens_one_key_field(should) if conjunctive else None
                        if widened is None:
                            disqualified[0] = True
                            return
                        seen.add(widened)
                        for occ, body in value.items():
                            if occ != "should":
                                walk(body, conjunctive and occ in ("must", "filter"))
                        continue
                    for occ, body in value.items():
                        # `must_not` is an exclusion: a field named only there was never
                        # asked "is this identity present", so it cannot make a 0-row
                        # result mean "absent".
                        walk(body, conjunctive and occ in ("must", "filter"))
                    continue
                if key in _DSL_FIELD_CLAUSES:
                    if conjunctive:
                        seen.update(
                            leaf_of(f).lower() for f in _dsl_clause_fields(node)
                        )
                    continue
                walk(value, conjunctive)
            return
        if isinstance(node, list):
            for item in node:
                walk(item, conjunctive)

    walk(dsl.get("query", dsl), True)
    if disqualified[0]:
        return False
    return wanted.issubset(seen)


def _form_leaf_groups(form_bindings: Optional[List[tuple]]) -> Dict[str, Set[str]]:
    """``{entity type: {leaf, ...}}`` from the pack's per-form column declaration.

    The input is :func:`field_mapping.form_split_bindings` verbatim, the same declaration
    :func:`relax_form_conjunction_dsl` reads, so the two agree about which columns belong to
    one type. This route survives a literal the value route cannot recognise (a generated
    prefix or pattern whose ends do not unwrap), which is why it is kept beside the value
    route rather than replaced by it.
    """
    out: Dict[str, Set[str]] = {}
    for entry in form_bindings or []:
        try:
            etype, forms = entry
        except (TypeError, ValueError):
            continue
        leaves: Set[str] = set()
        for form in forms or []:
            try:
                _name, columns = form
            except (TypeError, ValueError):
                continue
            for column in columns or []:
                leaf = leaf_of(str(column)).lower()
                if leaf:
                    leaves.add(leaf)
        if leaves:
            out.setdefault(str(etype), set()).update(leaves)
    return out


def _arm_entity_type(clause: Any, by_value: Dict[str, List[str]]) -> Optional[str]:
    """The single entity type every literal in this arm is a value of, else ``None``.

    Uses the same attribution as :func:`strip_fabricated_predicates` and
    :func:`enforce_conjunction_same_column_dsl` via :func:`_literal_value_types`. Refused in
    three ambiguous cases: a literal of no known type, literals of different types in one
    ``terms`` list, and a value shared by two types with nothing to break the tie.
    """
    pairs = _dsl_clause_pairs(clause)
    if not pairs:
        return None
    types: Optional[Set[str]] = None
    for _field, literal in pairs:
        found = set(_literal_value_types(literal, by_value))
        if not found:
            return None
        types = found if types is None else (types & found)
        if not types:
            return None
    return next(iter(types)) if types and len(types) == 1 else None


def _value_leaf_groups(
    clauses: List[Any], by_value: Dict[str, List[str]]
) -> Dict[str, Set[str]]:
    """``{entity type: {leaf, ...}}`` attributed from the literals each arm carries.

    The route the declaration cannot supply: a type bound flat to several columns declares no
    forms, and a column the generator reached for that the pack does not declare appears in no
    binding. See :func:`enforce_conjunction_dsl`.
    """
    out: Dict[str, Set[str]] = {}
    for clause in clauses:
        fields = _dsl_clause_fields(clause)
        if len(fields) != 1:
            continue
        etype = _arm_entity_type(clause, by_value)
        if not etype:
            continue
        leaf = leaf_of(str(fields[0])).lower()
        if leaf:
            out.setdefault(etype, set()).add(leaf)
    return out


def _merge_leaf_groups(*groups: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Union the two routes' groupings, dropping every leaf two types both claim.

    A blob column several types collapse onto is :func:`enforce_conjunction_same_column_dsl`'s
    subject; there the arms must be AND-ed, here they would be OR-ed, so guessing would
    produce the opposite result.
    """
    merged: Dict[str, Set[str]] = {}
    for group in groups:
        for etype, leaves in group.items():
            merged.setdefault(etype, set()).update(leaves)
    counts: Dict[str, int] = {}
    for leaves in merged.values():
        for leaf in leaves:
            counts[leaf] = counts.get(leaf, 0) + 1
    return {
        etype: {leaf for leaf in leaves if counts.get(leaf, 0) == 1}
        for etype, leaves in merged.items()
    }


def _sibling_leaves(groups: Dict[str, Set[str]], required: Set[str]) -> Dict[str, str]:
    """``{a sibling column's leaf: the key member's own leaf}``, per entity type.

    A group is used only when exactly one of its columns is a key member. Zero means the key
    does not name this type and its arms stay spare. Two or more means the key itself names two
    of this type's columns; folding them together would delete a conjunction the pack asked for,
    so the per-field reading stands.
    """
    out: Dict[str, str] = {}
    for leaves in groups.values():
        owned = sorted(leaf for leaf in leaves if leaf in required)
        if len(owned) != 1:
            continue
        for leaf in leaves:
            if leaf not in required:
                out[leaf] = owned[0]
    return out


#: Occurrence keys that make a ``bool`` a pure conjunction. ``should`` and ``must_not`` are
#: absent: an arm holding either is not a conjunction of its parts, so decomposing it would not
#: preserve what it selects.
_DSL_CONJUNCTION_KEYS = ("must", "filter")


def _dsl_conjunction_arms(clause: Any) -> Optional[List[Any]]:
    """The conjuncts of a clause that is a pure-conjunction ``bool``, else ``None``.

    A generator that has already conjoined two key members writes them as one nested
    ``bool.must``/``bool.filter`` arm, and :func:`_dsl_clause_fields` reads a ``bool`` as
    constraining no field. A conjunction over key members is narrower than every other arm, so
    it must not read as unclassifiable. Returns the conjuncts verbatim so the caller re-emits
    them rather than rebuilding them.

    Returns ``None`` for every non-unambiguous shape: a nested ``should``/``must_not``, a
    conjunct that is itself not a single-field leaf, or a ``bool`` carrying no conjunctive key.
    The caller must check which fields came back; a conjunct the key does not name would
    become a spare OR arm on decomposition, widening the result.
    """
    if not isinstance(clause, dict) or set(clause) != {"bool"}:
        return None
    body = clause["bool"]
    if not isinstance(body, dict):
        return None
    # Any key beyond the two conjunctive ones can change what the arm selects. `minimum_should_match`
    # is included in that refusal: a `bool` carrying only a `filter` and `minimum_should_match: 1`
    # matches no document, which is why `drop_orphan_minimum_should_match` runs terminally.
    if any(key not in _DSL_CONJUNCTION_KEYS for key in body):
        return None
    arms: List[Any] = []
    for key in _DSL_CONJUNCTION_KEYS:
        if key not in body:
            continue
        group = body[key]
        group = group if isinstance(group, list) else [group]
        for conjunct in group:
            # Exactly one field per conjunct, or it is not attributable per field and the
            # per-field grouping this feeds could not place it.
            if len(_dsl_clause_fields(conjunct)) != 1:
                return None
            arms.append(conjunct)
    return arms or None


def _expand_conjunction_arms(
    clauses: List[Any], required: Set[str], form_groups: Dict[str, Set[str]]
) -> List[Any]:
    """Flatten one already-conjoined key arm so the per-field grouping can read its leaves.

    A nested ``bool`` conjunction reads as no fields; flattening lets the sibling fold into
    its own type's conjunct. At most one arm is flattened: two would group per field into the
    cross product. Only conjuncts the key names are included; others would become spare arms.
    """
    allowed = set(required) | set(_sibling_leaves(form_groups, required))
    expandable = [
        index
        for index, clause in enumerate(clauses)
        if _is_flattenable_key_arm(clause, allowed)
    ]
    if len(expandable) != 1:
        return list(clauses)
    target = expandable[0]
    # Only where there is a bare key arm to fold in; a group whose every other arm is outside the
    # key is already scoped correctly (spare-arm reading), so flattening it would rewrite a correct
    # query into an equivalent one and turn a `filter` into a `must` for no row.
    if not any(
        index != target
        and (leaves := _dsl_clause_fields(clause))
        and len(leaves) == 1
        and leaf_of(leaves[0]).lower() in allowed
        for index, clause in enumerate(clauses)
    ):
        return list(clauses)
    out: List[Any] = []
    for index, clause in enumerate(clauses):
        if index == target:
            out.extend(_dsl_conjunction_arms(clause) or [clause])
        else:
            out.append(clause)
    return out


def _is_flattenable_key_arm(clause: Any, allowed: Set[str]) -> bool:
    """Is this arm a conjunction of >=2 leaves, every one of them a column the key reaches?"""
    arms = _dsl_conjunction_arms(clause)
    if not arms or len(arms) < 2:
        return False
    leaves = {leaf_of(_dsl_clause_fields(a)[0]).lower() for a in arms}
    return len(leaves) >= 2 and leaves <= allowed


def enforce_conjunction_dsl(
    dsl: Dict,
    fields: List[str],
    source_name: str = "?",
    form_bindings: Optional[List[tuple]] = None,
    incident_values: Optional[Dict[str, Iterable[str]]] = None,
) -> Dict:
    """Promote a ``bool.should`` over composite-key fields to ``bool.must``. Returns a new dict.

    Spare arms (not named by the key) are kept OR-ed beside the conjunction. Arms on the same
    field are OR-ed into one conjunct, never AND-ed. Grouping is per entity type, not per
    field. Arms are grouped from two routes: the pack's ``form_bindings`` (survives literals
    the value route cannot recognise), and :func:`_arm_entity_type` literal attribution.
    Every arm is re-emitted verbatim.
    """
    if not isinstance(dsl, dict) or len(fields or []) < 2:
        return dsl
    required = {leaf_of(f).lower() for f in fields}
    # Two spellings of one leaf collapse here; there is no conjunction between a field and
    # itself, so promoting that group would AND two values of one column.
    if len(required) < 2:
        return dsl
    form_groups = _form_leaf_groups(form_bindings)
    by_value = _types_by_value(incident_values or {})
    rewritten = [0]
    spared = [0]
    grouped = [0]
    folded = [0]

    def promote(value: Dict) -> Optional[Dict]:
        clauses = value["should"]
        clauses = clauses if isinstance(clauses, list) else [clauses]
        if len(clauses) < 2:
            return None
        # An arm that already conjoins key members is unreadable per field, and abandoning the
        # group on it leaves every bare sibling arm bounded by nothing: the full result this
        # guard exists to prevent, arriving because the generator got half of it right.
        clauses = _expand_conjunction_arms(clauses, required, form_groups)
        if len(clauses) < 2:
            return None
        per_clause = [
            [leaf_of(f).lower() for f in _dsl_clause_fields(c)] for c in clauses
        ]
        # A clause constraining no field at all (a `match_all`, a nesting this cannot read) is
        # not something to reason about: it may be wider than everything else in the group.
        if not all(per_clause):
            return None
        # {a sibling column's leaf: the key member's own leaf}: a type spread across several
        # columns is one conjunct, so its arms group under the leaf the key names. Resolved per
        # `should` group, because half the attribution reads the group's own literals.
        siblings = _sibling_leaves(
            _merge_leaf_groups(form_groups, _value_leaf_groups(clauses, by_value)),
            required,
        )

        def canon(leaf: str) -> str:
            return siblings.get(leaf, leaf)

        seen = {leaf for leaves in per_clause for leaf in leaves}
        if not required <= {canon(leaf) for leaf in seen}:
            return None
        if not all(len(leaves) == 1 for leaves in per_clause):
            # Unclassifiable per field: AND the clauses wholesale. Per-type canonicalisation
            # is not applied here because treating a sibling form's column as the key member's
            # would AND two forms of one subject.
            if seen != required:
                return None
            new_bool = {k: walk(v) for k, v in value.items() if k != "should"}
            must = new_bool.get("must", [])
            must = list(must) if isinstance(must, list) else [must]
            new_bool["must"] = must + [walk(c) for c in clauses]
            new_bool.pop("minimum_should_match", None)
            rewritten[0] += 1
            return new_bool
        by_field: Dict[str, List[Any]] = {}
        raw_leaves: Dict[str, Set[str]] = {}
        order: List[str] = []
        spare: List[Any] = []
        for leaves, clause in zip(per_clause, clauses):
            leaf = canon(leaves[0])
            if leaf not in required:
                spare.append(clause)
                continue
            if leaf not in by_field:
                by_field[leaf] = []
                raw_leaves[leaf] = set()
                order.append(leaf)
            by_field[leaf].append(clause)
            raw_leaves[leaf].add(leaves[0])
        conjuncts: List[Any] = []
        for leaf in order:
            arms = by_field[leaf]
            if len(arms) == 1:
                conjuncts.append(walk(arms[0]))
                continue
            # `minimum_should_match` is explicit because the default is 1 only while the bool
            # carries no `must`, and the promotion below puts one there.
            conjuncts.append(
                {
                    "bool": {
                        "should": [walk(a) for a in arms],
                        "minimum_should_match": 1,
                    }
                }
            )
            # Counted apart: several values on one column is the ordinary form list; arms folded
            # across a type's columns are the ones that would otherwise ride out unscoped.
            if len(raw_leaves[leaf]) > 1:
                folded[0] += 1
            else:
                grouped[0] += 1
        new_bool = {k: walk(v) for k, v in value.items() if k != "should"}
        if not spare:
            must = new_bool.get("must", [])
            must = list(must) if isinstance(must, list) else [must]
            new_bool["must"] = must + conjuncts
            new_bool.pop("minimum_should_match", None)
        else:
            # The conjunction is one alternative; promoting it into `must` beside the spare arm
            # would AND the two and demand they co-occur. Existing `must`/`filter` (window,
            # partition bound) is walked and never rebuilt.
            new_bool["should"] = [{"bool": {"must": conjuncts}}] + [
                walk(c) for c in spare
            ]
            new_bool["minimum_should_match"] = 1
            spared[0] += len(spare)
        rewritten[0] += 1
        return new_bool

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict) and "should" in value:
                    promoted = promote(value)
                    if promoted is not None:
                        out[key] = promoted
                        continue
                out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    if rewritten[0]:
        logger.warning(
            "Source '%s' is keyed by %s: rewrote %d should -> must in the generated query "
            "DSL (an OR on a composite-key lookup returns other keys' rows)%s%s%s.",
            source_name,
            " + ".join(leaf_of(f) for f in fields),
            rewritten[0],
            (
                f"; {spared[0]} arm(s) the key does not name kept OR-ed beside it"
                if spared[0]
                else ""
            ),
            (
                f"; {grouped[0]} key column(s) carried several values, OR-ed within the column"
                if grouped[0]
                else ""
            ),
            (
                f"; {folded[0]} key member(s) constrained on several of their own entity TYPE's "
                "columns, OR-ed inside the member's conjunct"
                if folded[0]
                else ""
            ),
        )
    return result


def enforce_conjunction_same_column_dsl(
    dsl: Dict,
    groups: List[tuple],
    incident_values: Dict[str, Iterable[str]],
    source_name: str = "?",
) -> Dict:
    """Promote a ``should`` over one column's required types to ``must``, grouped by type.

    Query DSL twin of :func:`enforce_conjunction_same_column`. Both resolve arms through
    :func:`_same_column_conjuncts`. Here every clause names the same field, so the arity
    check that gates the sibling cannot apply; what decides is the type of each arm's literal.

    Arms of one entity type stay OR-ed as a nested ``bool.should`` with an explicit
    ``minimum_should_match: 1``. The default is 1 only when the bool carries no ``must``;
    the promotion puts one there, so leaving it implicit would make every arm mandatory and
    two forms of one identity would have to co-occur in a document.
    """
    if not isinstance(dsl, dict) or not groups or not incident_values:
        return dsl
    by_value = _types_by_value(incident_values)
    if not by_value:
        return dsl
    applied: List[str] = []
    declined: List[str] = []

    def arm_of(clause: Any) -> Optional[tuple]:
        """``(field, literal, clause)`` if this leaf compares one field to one literal."""
        fields = _dsl_clause_fields(clause)
        pairs = _dsl_clause_pairs(clause)
        if len(fields) != 1 or len(pairs) != 1:
            return None
        field, literal = pairs[0]
        return field, literal, clause

    def promote(value: Dict) -> Optional[Dict]:
        clauses = value["should"]
        clauses = clauses if isinstance(clauses, list) else [clauses]
        if len(clauses) < 2:
            return None
        arms = [arm_of(c) for c in clauses]
        if any(a is None for a in arms):
            declined.append("a should clause is not a single field/single literal leaf")
            return None
        for field, members in groups:
            split = _same_column_conjuncts(arms, field, members, by_value)
            if split is None:
                continue
            conjuncts, spare = split
            new_bool = {k: walk(v) for k, v in value.items() if k != "should"}
            key_clauses: List[Any] = []
            for group in conjuncts:
                if len(group) == 1:
                    key_clauses.append(walk(group[0]))
                else:
                    key_clauses.append(
                        {
                            "bool": {
                                "should": [walk(c) for c in group],
                                "minimum_should_match": 1,
                            }
                        }
                    )
            note = f"{field} ({len(conjuncts)} of {'/'.join(members)})"
            if not spare:
                must = new_bool.get("must", [])
                must = list(must) if isinstance(must, list) else [must]
                new_bool["must"] = must + key_clauses
                # The outer `minimum_should_match` described a `should` list that no longer
                # exists; left behind it would be read against the promoted `must`.
                new_bool.pop("minimum_should_match", None)
                applied.append(note)
                return new_bool
            # With a spare arm, the conjunction is one alternative; the `should` stays a `should`.
            # Promoting it into `must` beside the spare would AND the two and demand co-occurrence.
            # Existing `must`/`filter` is walked and not rebuilt.
            new_bool["should"] = [{"bool": {"must": key_clauses}}] + [
                walk(c) for c in spare
            ]
            new_bool["minimum_should_match"] = 1
            applied.append(f"{note}, {len(spare)} clause(s) left beside it")
            return new_bool
        declined.append("no declared same-column key accounts for two or more should clauses")
        return None

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict) and "should" in value:
                    promoted = promote(value)
                    if promoted is not None:
                        out[key] = promoted
                        continue
                out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    for note in applied:
        logger.warning(
            "Source '%s': promoted a should(OR) to must(AND) on a SHARED column — %s. Every "
            "member of this source's declared key binds to that one column, so the tuple "
            "resolved to a single field and the ordinary conjunction guard had nothing to "
            "rewrite; the OR-ed form returns every document naming ANY member.",
            source_name,
            note,
        )
    if declined and not applied:
        logger.info(
            "Source '%s': left %d should-group(s) as generated on a shared key column — %s.",
            source_name,
            len(declined),
            "; ".join(sorted(set(declined))),
        )
    return result


def _dsl_terms_values(clause: Any) -> Optional[tuple]:
    """``(field, [value, ...])`` if ``clause`` is one plain equality/membership leaf.

    DSL twin of :func:`_column_literals`, conservative in the same way: the clause must be a
    single ``term``/``terms``/``match``/``match_phrase`` on one field and nothing else. A
    ``range`` is a bound, an ``exists`` carries no value, a ``prefix``/``wildcard``/``regexp``
    matches values this pass never harvested, and a clause carrying a boost or a nested ``bool``
    says more than this can account for; all make the caller leave the query as generated.
    """
    if not isinstance(clause, dict) or len(clause) != 1:
        return None
    key, body = next(iter(clause.items()))
    if key not in ("term", "terms", "match", "match_phrase"):
        return None
    if not isinstance(body, dict) or len(body) != 1:
        return None
    field, spec = next(iter(body.items()))
    if isinstance(spec, dict):
        # The long form: {"term": {"f": {"value": "x"}}}. Anything beside the value (a boost,
        # a case_insensitive flag, an analyzer) changes what the clause matches.
        if set(spec) - {"value", "query"} or not spec:
            return None
        spec = next(iter(spec.values()))
    if isinstance(spec, (list, tuple)):
        values = [str(v) for v in spec]
    elif spec is None or isinstance(spec, bool):
        return None
    else:
        values = [str(spec)]
    if not values:
        return None
    return str(field), values


def _dsl_slot_values(clause: Any) -> Optional[List[tuple]]:
    """The ``[(field, [value, ...]), ...]`` one ``must``/``filter`` clause constrains, or ``None``.

    DSL twin of :func:`_slot_columns`. A combination's components need not arrive as a leaf:
    ``relax_form_conjunction_dsl`` runs before this guard and may have replaced an AND across
    form columns with a nested ``bool.should``, so a leaf-only reader would resolve one
    component and the caller would decline for having fewer than two.

    A nested ``bool`` qualifies only as a plain OR: a ``should`` list, no
    ``must``/``filter``/``must_not``, ``minimum_should_match`` absent or exactly 1, and every arm
    a readable leaf. Anything else returns ``None`` and leaves the clause as generated.
    """
    parsed = _dsl_terms_values(clause)
    if parsed is not None:
        return [parsed]
    if not isinstance(clause, dict) or len(clause) != 1:
        return None
    key, body = next(iter(clause.items()))
    if key != "bool" or not isinstance(body, dict):
        return None
    if set(body) - {"should", "minimum_should_match"}:
        return None
    msm = body.get("minimum_should_match", 1)
    if msm not in (1, "1"):
        return None
    should = body.get("should")
    arms = should if isinstance(should, list) else [should]
    if not arms:
        return None
    out: List[tuple] = []
    for arm in arms:
        leaf = _dsl_terms_values(arm)
        if leaf is None:
            return None
        out.append(leaf)
    return out


def enforce_value_tuples_dsl(
    dsl: Dict,
    tuples: Sequence[Sequence[tuple]],
    source_name: str = "?",
) -> Dict:
    """Rewrite AND-ed per-field value lists into the combinations that actually occurred.

    Query DSL mirror of :func:`enforce_value_tuples`. The same four properties that make the
    textual rewrite strictly narrowing hold here; literals are spliced from the generated
    clauses rather than re-serialised. The family is chosen per ``must``/``filter`` list,
    only that shape's clauses are replaced, and the proof runs against the chosen family's
    own harvest.

    A ``should`` is not touched: its arms are already alternatives, and promoting one would
    invent a constraint. The component clauses are replaced by a single nested ``bool.should``
    of ``bool.filter`` arms with ``minimum_should_match: 1``, an OR of ANDs. Returns a new dict.
    """
    if not isinstance(dsl, dict) or not tuples:
        return dsl
    harvested: Dict[str, set] = {}
    for tup in tuples:
        for column, literals in tup:
            harvested.setdefault(leaf_of(column).lower(), set()).update(
                str(v).lower() for v in literals
            )
    if len(harvested) < 2:
        return dsl
    stats = {"rewritten": 0, "arms": 0, "product": 0}

    def rewrite_list(clauses: List[Any]) -> Optional[List[Any]]:
        """The clause list with the components collapsed into one OR-of-ANDs, or ``None``."""
        positions: Dict[str, int] = {}
        asked: Dict[str, Dict[str, Any]] = {}
        for index, clause in enumerate(clauses):
            parsed = _dsl_slot_values(clause)
            if parsed is None:
                continue
            leaves = [leaf_of(field).lower() for field, _values in parsed]
            if any(leaf not in harvested or leaf in positions for leaf in leaves):
                continue
            # All-or-nothing per clause; re-emitted verbatim; refusal leaves the slot unclaimed
            # and declines the whole rewrite.
            for leaf, (field, values) in zip(leaves, parsed):
                positions[leaf] = index
                for value in values:
                    asked.setdefault(leaf, {}).setdefault(
                        str(value).lower(), []
                    ).append((field, value))
        # The clause index is the slot; two fields inside one `should` are alternatives, not
        # two components. See :func:`_rewrite_tuple_body`.
        slots: Dict[int, List[str]] = {}
        for leaf, index in positions.items():
            slots.setdefault(index, []).append(leaf)
        if len(positions) < 2 or len(slots) < 2:
            return None
        # One component resolves to the (field, value) pairs the clause asked it on; several
        # only where two spellings of one leaf were OR-ed together.
        members: List[Dict[str, List[tuple]]] = []
        for tup in tuples:
            by_leaf: Dict[str, List[tuple]] = {}
            for column, literals in tup:
                leaf = leaf_of(column).lower()
                if leaf not in positions:
                    continue
                for value in literals:
                    pairs = asked[leaf].get(str(value).lower())
                    if pairs is not None:
                        by_leaf[leaf] = pairs
                        break
            if by_leaf:
                members.append(by_leaf)
        shapes: Dict[frozenset, int] = {}
        for by_leaf in members:
            covered = frozenset(
                index
                for index, leaves in slots.items()
                if any(leaf in by_leaf for leaf in leaves)
            )
            if len(covered) > 1:  # one slot is not a combination across slots
                shapes[covered] = shapes.get(covered, 0) + 1
        families: Dict[frozenset, List[Dict[str, List[tuple]]]] = {}
        for shape in shapes:
            family: List[Dict[str, List[tuple]]] = []
            for by_leaf in members:
                kept = {
                    leaf: pairs
                    for leaf, pairs in by_leaf.items()
                    if positions[leaf] in shape
                }
                if all(any(leaf in kept for leaf in slots[index]) for index in shape):
                    family.append(kept)
            families[shape] = family
        if not families:
            logger.warning(
                "Source '%s': the generated query DSL constrains %s but none of the %d "
                "harvested combination(s) names two of those slots together, so it was left as "
                "written — there is no combination ACROSS slots to narrow to.",
                source_name,
                " + ".join(sorted(positions)),
                len(tuples),
            )
            return None
        chosen = max(
            families,
            key=lambda shape: (len(families[shape]), len(shape), [-i for i in sorted(shape)]),
        )
        family = families[chosen]
        for shape, count in sorted(shapes.items(), key=lambda kv: sorted(kv[0])):
            if shape == chosen:
                continue
            logger.info(
                "Source '%s': %d harvested combination(s) name a different shape (%s) and did "
                "not narrow this query DSL — the %d combination(s) of shape (%s) did. Every "
                "clause outside that shape keeps the predicate the generator wrote.",
                source_name,
                count,
                " + ".join(sorted(leaf for index in shape for leaf in slots[index])),
                len(family),
                " + ".join(sorted(leaf for index in chosen for leaf in slots[index])),
            )
        # The proof, over the slots this rewrite replaces and against the chosen family's own
        # harvest. Family-relative: a value harvested only by a discarded combination appears
        # in no arm, so the global harvest would pass and the rewrite would delete that value.
        chosen_leaves = [leaf for index in chosen for leaf in slots[index]]
        family_values: Dict[str, set] = {leaf: set() for leaf in chosen_leaves}
        for by_leaf in family:
            # `asked[leaf]` is keyed by value, so a component's several fields are one entry
            # and all carry the same value; the proof is about values, not fields.
            for leaf, pairs in by_leaf.items():
                family_values[leaf].add(str(pairs[0][1]).lower())
        for leaf in chosen_leaves:
            extra = set(asked[leaf]) - family_values[leaf]
            if extra:
                logger.warning(
                    "Source '%s': left the generated query DSL as written — field %s carries "
                    "value(s) (%s) that this pass did not harvest, so collapsing the AND-ed "
                    "value lists into the observed combinations could not be shown to narrow "
                    "the query. Their cross product is asked as generated.",
                    source_name,
                    leaf,
                    ", ".join(sorted(extra)),
                )
                return None
        arms: List[Any] = []
        seen: set = set()
        for by_leaf in family:
            # One component's spellings all carry the same value (one `asked` entry), so
            # arm identity is its (component, value) set, not its field set.
            key = tuple(
                sorted(
                    (leaf, str(pairs[0][1]).lower()) for leaf, pairs in by_leaf.items()
                )
            )
            if key in seen:
                continue
            seen.add(key)
            # Clause order as generated, inside the arm as well; see :func:`_rewrite_tuple_body`.
            filters: List[Any] = []
            for index in sorted(chosen):
                present = [leaf for leaf in slots[index] if leaf in by_leaf]
                # One term per field the slot asked this value on, all alternatives. AND-ing
                # them would claim that one row carries the value on every field, which no
                # guard may assert.
                terms = [
                    {"term": {field: value}}
                    for leaf in present
                    for field, value in by_leaf[leaf]
                ]
                filters.append(
                    terms[0]
                    if len(terms) == 1
                    else {"bool": {"should": terms, "minimum_should_match": 1}}
                )
            arms.append({"bool": {"filter": filters}})
        if not arms:
            logger.warning(
                "Source '%s': the generated query DSL constrains %s but NONE of the %d "
                "harvested combinations is fully within the values it asked for, so it was "
                "left as written — every row it can return is a pairing that did not occur.",
                source_name,
                " + ".join(sorted(positions)),
                len(tuples),
            )
            return None
        # Product across slots, sum within one, over the chosen shape only. A slot this
        # rewrite leaves alone is not part of what it narrowed.
        product = 1
        for index in chosen:
            product *= sum(len(asked[leaf]) for leaf in slots[index])
        if len(arms) >= product:
            return None
        replaced = set(chosen)
        first = min(replaced)
        group = {"bool": {"should": arms, "minimum_should_match": 1}}
        out: List[Any] = []
        for index, clause in enumerate(clauses):
            if index == first:
                out.append(group)
            elif index not in replaced:
                out.append(clause)
        stats["rewritten"] += 1
        stats["arms"] += len(arms)
        stats["product"] += product
        return out

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict):
                    new_bool: Dict[str, Any] = {}
                    for bkey, bvalue in value.items():
                        if bkey in ("must", "filter"):
                            clauses = bvalue if isinstance(bvalue, list) else [bvalue]
                            # Children first; the output is not re-walked. Bottom-up so a nested
                            # rewrite cannot be re-read as a component.
                            walked = [walk(c) for c in clauses]
                            collapsed = rewrite_list(walked)
                            if collapsed is not None:
                                new_bool[bkey] = collapsed
                                continue
                            new_bool[bkey] = (
                                walked if isinstance(bvalue, list) else walked[0]
                            )
                        else:
                            new_bool[bkey] = walk(bvalue)
                    out[key] = new_bool
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    if stats["rewritten"]:
        logger.warning(
            "Source '%s': collapsed %d AND-ed value list group(s) in the generated query DSL "
            "into the %d combination(s) actually observed — the cross product asked for %d, so "
            "%d pairing(s) belonging to other parties were dropped from the query's key space.",
            source_name,
            stats["rewritten"],
            stats["arms"],
            stats["product"],
            stats["product"] - stats["arms"],
        )
    return result


def enforce_identity_scope_dsl(
    dsl: Dict,
    scope_fields: List[str],
    synonym_fields: Optional[Any] = None,
    source_name: str = "?",
) -> Dict:
    """Split a flat ``bool.should`` into ``must`` conjuncts: OR inside a family, AND between.

    DSL twin of :func:`enforce_identity_scope`. Two properties keep it safe:

    * A family must be complete to be enforced (see :func:`_group_by_family`): a family
      generated for only a subset of a source's document shapes would delete the rows of the
      shapes it missed.
    * Nothing is ever added. Only clauses the generator already wrote are moved. A family the
      query never constrained is not invented, and a ``should`` carrying any clause on an
      undeclared field is left exactly as generated.

    Returns a new dict; the input is not mutated.
    """
    if not isinstance(dsl, dict):
        return dsl
    families = synonym_families(synonym_fields)
    # Matched on the full path, not the leaf. A DSL field name is its complete path, and leaves
    # collide in both directions: two shapes of one family can spell the same last segment, while
    # another family's column may end in a segment as generic as `value`.
    scopes = [str(f).lower() for f in (scope_fields or []) if f]
    fam_leaves = [[str(f).lower() for f in fam] for fam in families]
    # One family and no scope is the shape a plain OR-group already answers correctly.
    if not scopes and len(fam_leaves) < 2:
        return dsl
    known = set(scopes) | {leaf for fam in fam_leaves for leaf in fam}
    rewritten: List[str] = []

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "bool" and isinstance(value, dict) and "should" in value:
                    rebuilt = _split_identity_should(
                        value, scopes, fam_leaves, known, source_name, rewritten
                    )
                    if rebuilt is not None:
                        out[key] = {k: walk(v) for k, v in rebuilt.items()}
                        continue
                out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    for line in rewritten:
        logger.warning(
            "Source '%s': rewrote a flat identity should-group into %s — OR-ing a SCOPE (a "
            "unit, a role) or a SECOND independent identifier onto the actor's own asks for "
            "every row sharing either, and under the row cap the subject's own rows need not "
            "be in what comes back.",
            source_name,
            line,
        )
    if not rewritten and _has_should(dsl):
        logger.info(
            "Source '%s': identity families %s declared, but no rewritable flat should-group "
            "was found — the query DSL is left exactly as generated.",
            source_name,
            " / ".join("+".join(f) for f in fam_leaves) or ", ".join(scopes),
        )
    return result


def enforce_subject_anchor_dsl(
    dsl: Dict,
    identity_values: Dict[str, List[str]],
    scope_values: Dict[str, List[str]],
    source_name: str = "?",
) -> Dict:
    """Add the subject's identity to a Query DSL query that constrains it nowhere.

    DSL twin of :func:`enforce_subject_anchor`. The original query is nested as one clause of
    a new ``bool.filter`` alongside the anchor, preserving its own boolean structure. The
    identity columns become one nested ``bool.should`` with ``minimum_should_match: 1``; each
    scope is a ``terms`` beside it, AND-ed.

    Presence is tested loosely: the field name appearing anywhere in the encoded query. Errs
    toward "already constrained", i.e. toward leaving the generator's query alone.
    Returns a new dict.
    """
    if not isinstance(dsl, dict) or not identity_values:
        return dsl
    encoded = json.dumps(dsl)
    for col in list(identity_values) + list(scope_values or {}):
        if f'"{col}"' in encoded:
            logger.info(
                "Source '%s': the generated query DSL already names %s, so the subject "
                "anchor is not injected.",
                source_name,
                col,
            )
            return dsl
    should = [{"terms": {col: list(vals)}} for col, vals in identity_values.items() if vals]
    if not should:
        return dsl
    clauses: List[Dict[str, Any]] = [
        should[0]
        if len(should) == 1
        else {"bool": {"should": should, "minimum_should_match": 1}}
    ]
    clauses += [
        {"terms": {col: list(vals)}} for col, vals in (scope_values or {}).items() if vals
    ]
    logger.warning(
        "Source '%s': injected the subject anchor on %s into the generated query DSL — it "
        "constrained the incident's identity on no field, so it asked about a population "
        "rather than about this subject, and under the row cap the subject's own rows need "
        "not have been in the page at all.",
        source_name,
        ", ".join(list(identity_values) + list(scope_values or {})),
    )
    return {"bool": {"filter": clauses + [copy.deepcopy(dsl)]}}


def enforce_key_presence_dsl(
    dsl: Dict,
    key_values: "Dict[str, Dict[str, List[str]]]",
    source_name: str = "?",
) -> Dict:
    """Add a resolved key member a Query DSL query constrains nowhere.

    DSL twin of :func:`enforce_key_presence`. The original query is nested under
    ``bool.filter``, and each missing entity type becomes a ``bool.should`` AND-ed beside it.
    ``minimum_should_match: 1`` is explicit: the default is 1 only while the ``bool`` carries
    no ``must``, and nesting under ``filter`` puts one there.

    Presence is tested by field name anywhere in the encoded query; errs toward "constrained".
    Returns a new dict.
    """
    if not isinstance(dsl, dict) or not key_values:
        return dsl
    encoded = json.dumps(dsl)
    missing = {}
    for entity, columns in key_values.items():
        columns = {c: v for c, v in (columns or {}).items() if c and v}
        if not columns:
            continue
        if any(f'"{col}"' in encoded for col in columns):
            continue
        missing[entity] = columns
    if not missing:
        return dsl
    clauses: List[Dict[str, Any]] = []
    for columns in missing.values():
        should = [{"terms": {col: list(vals)}} for col, vals in columns.items()]
        clauses.append(
            should[0]
            if len(should) == 1
            else {"bool": {"should": should, "minimum_should_match": 1}}
        )
    logger.warning(
        "Source '%s': injected the key member(s) %s into the generated query DSL — this "
        "source declares the actor key %s and the query constrained those member(s) on no "
        "field, so it asked about a population rather than about this key.",
        source_name,
        "; ".join(f"{e}=({', '.join(cols)})" for e, cols in missing.items()),
        " + ".join(key_values),
    )
    return {"bool": {"filter": clauses + [copy.deepcopy(dsl)]}}


def _has_should(node: Any) -> bool:
    """Does this DSL tree contain a ``bool.should`` anywhere? (for the not-rewritten log)"""
    if isinstance(node, dict):
        if isinstance(node.get("bool"), dict) and "should" in node["bool"]:
            return True
        return any(_has_should(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_should(v) for v in node)
    return False


def _split_identity_should(
    bool_body: Dict[str, Any],
    scopes: List[str],
    fam_leaves: List[List[str]],
    known: set,
    source_name: str,
    rewritten: List[str],
) -> Optional[Dict[str, Any]]:
    """Rebuild one ``bool`` whose ``should`` flattens several identity facts, or ``None``.

    ``None`` means "leave this node exactly as generated"; every bail-out below is a shape
    this function declines to restructure rather than one it handles approximately.
    """
    clauses = bool_body.get("should")
    clauses = clauses if isinstance(clauses, list) else [clauses]
    if len(clauses) < 2:
        return None
    # Each clause must be a single-field leaf on a declared field. A clause on anything else
    # (or a nested bool) means this group is not the flat identity OR this function can split.
    tagged: List[tuple] = []
    for clause in clauses:
        fields = _dsl_clause_fields(clause)
        if len(fields) != 1:
            return None
        leaf = str(fields[0]).lower()
        if leaf not in known:
            return None
        tagged.append((leaf, clause))

    groups, partial = _group_by_family(
        [(leaf, c) for leaf, c in tagged if leaf not in set(scopes)],
        fam_leaves,
        source_name,
    )
    scope_clauses = [c for leaf, c in tagged if leaf in set(scopes)]
    if not groups:
        # Nothing identifies the subject any more: either the group was all scopes, or every
        # family in it was partial. Enforcing only the scopes is the whole-population query
        # this guard exists to prevent, so leave the query exactly as generated.
        return None
    # One family, no scope and nothing dropped is what a plain `should` already means.
    if not scope_clauses and len(groups) < 2 and not partial:
        return None

    conjuncts: List[Any] = []
    for group in groups:
        conjuncts.append(
            group[0]
            if len(group) == 1
            else {"bool": {"should": group, "minimum_should_match": 1}}
        )
    conjuncts.extend(scope_clauses)

    new_bool = {k: v for k, v in bool_body.items() if k != "should"}
    existing = new_bool.get("must", [])
    existing = existing if isinstance(existing, list) else [existing]
    new_bool["must"] = existing + conjuncts
    # `minimum_should_match` for the replaced `should` group is removed. Any surviving
    # `should` is one this function did not touch; its setting rides along untouched.
    # Leaving the key would apply it to an absent clause list.
    new_bool.pop("minimum_should_match", None)
    # Full paths, not leaves: the leaves collide, and a log line of `sign OR sign OR sign`
    # is unreadable. This line is the only record of how the query was restructured.
    rewritten.append(
        " AND ".join(
            "(" + " OR ".join(_dsl_clause_fields(c)[0] for c in g) + ")"
            if len(g) > 1
            else _dsl_clause_fields(g[0])[0]
            for g in groups
        )
        + (
            " AND " + " AND ".join(_dsl_clause_fields(c)[0] for c in scope_clauses)
            if scope_clauses
            else ""
        )
    )
    return new_bool


def enforce_partition_bounds_dsl(
    dsl: Dict,
    partitions: Sequence[Dict[str, Any]],
    date_from: Any,
    date_to: Any,
    source_name: str = "?",
) -> Dict:
    """AND a bound on every unbounded partition column into a Query DSL query.

    The DSL analogue of :func:`enforce_partition_bounds`. Because there is no query TEXT to
    re-associate, the shape is unambiguous: the original query is nested as one clause of a
    new ``bool.filter`` alongside the bounds, which preserves its own boolean structure
    exactly. ``role: date`` becomes a ``range``, a calendar-part role becomes a ``terms``.
    Returns a new dict.
    """
    if not isinstance(dsl, dict) or not partitions:
        return dsl
    encoded = json.dumps(dsl)
    clauses: List[Dict[str, Any]] = []
    names: List[str] = []
    for bound in partition_bounds(partitions, date_from, date_to):
        name = bound["name"]
        # Already constrained anywhere in the query? Leave the generator's bound alone.
        if f'"{name}"' in encoded:
            continue
        if bound["kind"] == "in":
            clauses.append({"terms": {name: bound["values"]}})
        else:
            clauses.append(
                {"range": {name: {"gte": bound["low"], "lte": bound["high"]}}}
            )
        names.append(name)
    if not clauses:
        return dsl
    logger.warning(
        "Source '%s': injected a partition bound on %s into the generated query DSL — it "
        "was left unbounded, which reads every partition instead of the window's.",
        source_name,
        ", ".join(names),
    )
    return {"bool": {"filter": clauses + [copy.deepcopy(dsl)]}}


def enforce_epoch_window_dsl(
    dsl: Dict,
    time_columns: Sequence[Dict[str, Any]],
    date_from: Any,
    date_to: Any,
    source_name: str = "?",
) -> Dict:
    """Repair epoch-column ``range`` clauses in a Query DSL query; DSL analogue of
    :func:`enforce_epoch_window`.

    A ``range`` in ``bool.filter`` is AND-ed with everything, so a wrong one zeroes the
    result regardless of what the ``should`` clauses matched. Same conservatism as the
    textual guard: a clause whose interval overlaps the incident window is untouched; a
    date/ISO literal on an epoch column is always converted; a clause that misses the window
    entirely is replaced with the window. ``format``/``time_zone`` keys are dropped on a
    rewritten clause. Returns a new dict.
    """
    if not isinstance(dsl, dict) or not time_columns:
        return dsl
    specs = {
        leaf_of(s["name"]).lower(): s
        for s in epoch_specs(time_columns, date_from, date_to)
    }
    if not specs:
        return dsl
    repaired: List[str] = []

    def repair_range(field: str, body: Any, spec: Dict[str, Any]) -> Any:
        if not isinstance(body, dict):
            return body
        low, high, scale = spec["low"], spec["high"], spec["scale"]
        bounds: Dict[str, Any] = {}
        needs_fix = False
        for key in ("gte", "gt", "lte", "lt", "eq"):
            if key not in body:
                continue
            upper = key in ("lte", "lt")
            value = body[key]
            as_int = _as_epoch_int(value)
            if as_int is None:
                converted = _date_as_epoch(value, scale, upper)
                if converted is None:
                    return body  # not a shape we can reason about; leave it alone
                bounds[key] = converted
                needs_fix = True
            else:
                bounds[key] = as_int
        if not bounds:
            return body
        lo_eff = max(
            [v for k, v in bounds.items() if k in ("gte", "gt", "eq")] or [low]
        )
        hi_eff = min(
            [v for k, v in bounds.items() if k in ("lte", "lt", "eq")] or [high]
        )
        if not needs_fix and lo_eff <= high and hi_eff >= low:
            return body  # overlaps the window; the generator's bound stands
        original = ", ".join(f"{k}={body[k]}" for k in body if k in bounds)
        repaired.append(f"{field} range({original})")
        if needs_fix and lo_eff <= high and hi_eff >= low:
            # The literals were dates but the interval they describe is right: keep the
            # generator's intent, just expressed in the unit the column actually stores.
            fixed = {k: v for k, v in bounds.items()}
        else:
            fixed = {"gte": low, "lte": high}
        return {
            **{
                k: v
                for k, v in body.items()
                if k not in bounds and k not in ("format", "time_zone")
            },
            **fixed,
        }

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "range" and isinstance(value, dict):
                    out[key] = {
                        field: (
                            repair_range(field, body, specs[leaf_of(field).lower()])
                            if leaf_of(field).lower() in specs
                            else body
                        )
                        for field, body in value.items()
                    }
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result = walk(copy.deepcopy(dsl))
    if repaired:
        logger.warning(
            "Source '%s': rewrote %d epoch range clause(s) in the generated query DSL that "
            "could not match the incident's window (%s) — the range sits in bool.filter, so "
            "a wrong one zeroes the result no matter what the should clauses matched.",
            source_name,
            len(repaired),
            "; ".join(repaired),
        )
    return result


def partition_clauses_encoded(
    partitions: Sequence[Dict[str, Any]],
    date_from: Any,
    date_to: Any,
    already: Iterable[str] = (),
) -> List[str]:
    """Bounds as ServiceNow-style encoded-query clauses (``field>=value`` / ``fieldINa,b``).

    For a backend whose query is assembled in code rather than generated, the clauses are
    simply added. ``already`` names fields the caller has bounded itself, which are skipped.
    Returns ``[]`` when nothing needs adding.
    """
    done = {leaf_of(f).lower() for f in already or ()}
    out: List[str] = []
    for bound in partition_bounds(partitions, date_from, date_to):
        name = bound["name"]
        if leaf_of(name).lower() in done:
            continue
        if bound["kind"] == "in":
            out.append(f"{name}IN{','.join(bound['values'])}")
        else:
            out.append(f"{name}>={bound['low']}")
            out.append(f"{name}<={bound['high']}")
    return out


def epoch_clauses_encoded(
    time_columns: Sequence[Dict[str, Any]],
    date_from: Any,
    date_to: Any,
    already: Iterable[str] = (),
) -> List[str]:
    """Epoch bounds as encoded-query clauses (``field>=123``), for a query built in code.

    There is no generated text to repair on such a backend, so the clauses are simply added.
    Wired anyway: a pack field that silently no-ops on one backend is the class of defect
    these guards exist to prevent. ``already`` names fields the caller bounds itself, skipped.
    """
    done = {leaf_of(f).lower() for f in already or ()}
    out: List[str] = []
    for spec in epoch_specs(time_columns, date_from, date_to):
        if leaf_of(spec["name"]).lower() in done:
            continue
        out.append(f"{spec['name']}>={spec['low']}")
        out.append(f"{spec['name']}<={spec['high']}")
    return out


def guard_prompt_line(never_filter: Iterable[str]) -> str:
    """One prompt sentence stating the ``never_filter`` prohibition, or ``''``.

    The post-generation strip is the actual guarantee, but telling the generator up front
    avoids the rewrite (and the rewrite cannot fix every shape; a predicate buried in an
    OR-group is only warned about). Derived from the pack field so declaring
    ``never_filter`` is self-sufficient: no source has to hand-write the same prose into
    its ``query_hints``, where it would drift or be forgotten.
    """
    fields = [f for f in (never_filter or []) if f]
    if not fields:
        return ""
    return (
        "EVIDENCE FIELDS — "
        + ", ".join(str(f) for f in fields)
        + ": these are the ANSWER, not noise. RETURN them in the projection, and NEVER put "
        "them in a filter/WHERE condition. Filtering on one deletes exactly the rows that "
        "the investigation needs to inspect. This overrides any general guidance about "
        "excluding noisy or service-account rows."
    )


def required_fields(
    field_map: Dict[str, str], require_all_entities: Optional[Iterable[str]]
) -> List[str]:
    """Resolve ``require_all_entities`` entity types to backend fields, in order.

    Shared by every retriever so "which fields must be AND-ed" is computed one way.
    """
    out: List[str] = []
    for entity_type in _entity_types(require_all_entities):
        field = (field_map or {}).get(entity_type)
        if field and field not in out:
            out.append(field)
    return out


def same_column_conjunctions(
    field_map: Dict[str, str],
    require_all_entities: Optional[Iterable[str]] = None,
    identity_keys: Optional[Iterable[Iterable[str]]] = None,
    present_types: Optional[Iterable[str]] = None,
    source_name: str = "?",
) -> List[tuple]:
    """``[(field, [entity type, ...])]`` for every column carrying 2+ members of the key.

    Input to :func:`enforce_conjunction_same_column` and its DSL twin. Not routed through
    :func:`conjunction_fields`: that function chooses one declaration and returns distinct
    fields; this one needs the types from any declaration that names a conjunction, including
    candidates the key resolver skipped. Groups are returned in declaration order and deduped
    per ``(field, types)``.

    ``require_all_entities`` is read unconditionally. Of ``identity_keys`` only the first
    fully satisfied candidate is taken: lower-priority candidates are not additional requirements.
    """
    have = set(present_types) if present_types is not None else None
    declarations: List[List[str]] = []
    fixed = _entity_types(require_all_entities)
    if len(fixed) >= 2:
        declarations.append(fixed)
    for candidate in identity_keys or []:
        types = _entity_types(candidate)
        if len(types) < 2:
            continue
        if have is not None and not all(t in have for t in types):
            continue
        if all((field_map or {}).get(t) for t in types):
            declarations.append(types)
            break
    out: List[tuple] = []
    seen = set()
    for types in declarations:
        by_field: Dict[str, List[str]] = {}
        order: List[str] = []
        for entity_type in types:
            field = (field_map or {}).get(entity_type)
            if not field:
                continue
            if field not in by_field:
                by_field[field] = []
                order.append(field)
            if entity_type not in by_field[field]:
                by_field[field].append(entity_type)
        for field in order:
            members = by_field[field]
            if len(members) < 2:
                continue
            key = (field, tuple(members))
            if key in seen:
                continue
            seen.add(key)
            out.append((field, members))
    if out:
        logger.info(
            "Source '%s': %s — an OR between them on that column returns every record "
            "naming ANY of them, so it is rewritten to an AND after generation.",
            source_name,
            "; ".join(f"{f} binds {'/'.join(m)} at once" for f, m in out),
        )
    return out


def _entity_types(declared: Any) -> List[str]:
    """The entity types a pack declaration names: non-empty strings, order kept, deduped.

    Non-string members are dropped here rather than at the field lookup: a nested list passed
    to ``{}.get(...)`` raises ``TypeError`` inside a guard. A bare string yields nothing rather
    than its characters: ``identity_keys: [type_a]`` would otherwise resolve single characters.
    """
    if isinstance(declared, (str, bytes)) or isinstance(declared, dict):
        return []
    try:
        items = list(declared or [])
    except TypeError:
        return []
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if name and name not in out:
            out.append(name)
    return out


def resolve_identity_key(
    field_map: Dict[str, str],
    identity_keys: Optional[Iterable[Iterable[str]]],
    present_types: Optional[Iterable[str]] = None,
) -> List[str]:
    """Pick the first candidate actor key this incident and source can both satisfy.

    Returns the resolved backend fields of the first fully usable candidate: every member
    mapped to a real field on this source and carried by this incident. Returns ``[]`` when
    none qualifies.

    Candidates are resolved in two tiers: conjunctions first, then single-member candidates.
    A complete one-column key must not preempt a two-column one. A one-member candidate is a
    valid lookup key; ``key_was_enforced`` can claim an empty result via the superset rule.

    ``present_types`` are the entity types the incident carries. When omitted, only the field
    map constrains the choice.
    """
    have = set(present_types) if present_types is not None else None
    candidates = [_entity_types(c) for c in (identity_keys or [])]
    for conjunctions in (True, False):
        for types in candidates:
            if not types or (len(types) >= 2) is not conjunctions:
                continue
            if have is not None and not all(t in have for t in types):
                continue
            fields: List[str] = []
            for entity_type in types:
                field = (field_map or {}).get(entity_type)
                if not field:
                    fields = []
                    break
                if field not in fields:
                    fields.append(field)
            # A conjunction still needs two columns: two declared members that this source
            # binds to the same one cannot be AND-ed (`col = A AND col = B` matches nothing),
            # so that candidate falls through to the next one exactly as before.
            if len(fields) >= (2 if conjunctions else 1):
                return fields
    return []


def conjunction_fields(
    field_map: Dict[str, str],
    require_all_entities: Optional[Iterable[str]] = None,
    identity_keys: Optional[Iterable[Iterable[str]]] = None,
    present_types: Optional[Iterable[str]] = None,
    source_name: str = "?",
) -> List[str]:
    """The fields that must be AND-ed for this source, from either declaration.

    One function so the prompt hint and the post-generation rewrite always agree. The two
    declarations are unioned: ``identity_keys`` is priority-ordered conditional alternatives;
    ``require_all_entities`` is unconditional. A satisfied candidate and the fixed tuple are
    both AND-ed. A conjunction outranks a one-column key: two AND-ed columns bound the query
    to one actor; one column only licenses an empty result as "absent".
    """
    chosen = resolve_identity_key(field_map, identity_keys, present_types)
    # A non-empty return says which fields will be AND-ed, and nothing more. Whether the
    # rewrite landed them in the query is separate (`enforce_conjunction` leaves shapes it
    # cannot safely rewrite); a caller inferring "lookup was keyed" must read what was enforced.
    fixed = required_fields(field_map, require_all_entities)
    # Whichever declaration already named a conjunction keeps its own order; the other's extra
    # members are appended. Order is what an operator reads in the published predicate. A
    # one-member candidate must not displace the head of a fixed tuple of two.
    if len(chosen) >= 2:
        base, added = chosen, fixed
    else:
        base, added = fixed, chosen
    combined = base + [f for f in added if f not in base]
    if len(combined) >= 2:
        if chosen and len(combined) > len(chosen) == len(base):
            logger.info(
                "Source '%s': actor identified by %s (first satisfied identity_keys "
                "candidate), AND-ed with %s from require_all_entities — %s in all.",
                source_name,
                " + ".join(chosen),
                " + ".join(f for f in combined if f not in chosen),
                " + ".join(combined),
            )
        elif len(chosen) >= 2:
            logger.info(
                "Source '%s': actor identified by %s (first satisfied identity_keys "
                "candidate); these are AND-ed.",
                source_name,
                " + ".join(chosen),
            )
        elif chosen:
            logger.info(
                "Source '%s': require_all_entities resolved %s, AND-ed with the one-column "
                "identity_keys candidate %s — %s in all.",
                source_name,
                " + ".join(fixed),
                " + ".join(chosen),
                " + ".join(combined),
            )
        return combined
    if chosen:
        # A one-column key: nothing to AND. Returned because it is the field a reader of an
        # empty result needs (`key_was_enforced`); logging it keeps a one-member declaration
        # distinct from a satisfied pair in the log.
        logger.info(
            "Source '%s': actor key is the single column %s (first satisfied identity_keys "
            "candidate). Nothing is AND-ed — this only licenses reading an empty result as "
            "'this identity is not there' rather than as missing data.",
            source_name,
            chosen[0],
        )
        return chosen
    if identity_keys:
        # The pack declared candidate keys and none could be met; no conjunction will be
        # enforced. Which of three causes it was decides who can fix it, so the sentence
        # names the cause. Reported whatever the fallback resolved.
        logger.warning(
            "Source '%s': %s, and the require_all_entities fallback resolved %s — the "
            "query will NOT be constrained to a single actor.",
            source_name,
            identity_key_diagnosis(field_map, identity_keys, present_types),
            " + ".join(fixed) if fixed else "nothing",
        )
    elif len(fixed) == 1:
        # One-member `require_all_entities`: nothing to AND. Logged the same way as a
        # one-member `identity_keys` so the two are indistinguishable in the log.
        logger.info(
            "Source '%s': actor key is the single column %s (require_all_entities resolved "
            "one member). Nothing is AND-ed — this only licenses reading an empty result "
            "as 'this identity is not there' rather than as missing data.",
            source_name,
            fixed[0],
        )
    return fixed


def identity_key_diagnosis(
    field_map: Dict[str, str],
    identity_keys: Optional[Iterable[Iterable[str]]] = None,
    present_types: Optional[Iterable[str]] = None,
) -> str:
    """Why no ``identity_keys`` candidate resolved; three causes, three different owners.

    Read only after :func:`resolve_identity_key` came back empty. The three answers are not
    interchangeable: (1) every candidate is malformed and no incident can ever satisfy it:
    a catalog defect; (2) this source binds no column for a member the incident did carry:
    a stale or absent ``entity_bindings`` entry; (3) the incident genuinely carries only part
    of every candidate, not a defect.

    Prose only; nothing branches on the return value.
    """
    candidates = [_entity_types(c) for c in (identity_keys or [])]
    if not any(candidates):
        return (
            "every declared identity_keys candidate names no usable entity type at all "
            "(a candidate must be a LIST of entity-type names), so no incident can ever "
            "satisfy this declaration — it is a catalog defect, not a property of this "
            "incident"
        )
    have = set(present_types) if present_types is not None else None
    for types in candidates:
        if not types or (have is not None and not all(t in have for t in types)):
            continue
        unbound = [t for t in types if not (field_map or {}).get(t)]
        if unbound:
            return (
                "identity_keys candidate "
                + " + ".join(types)
                + " IS carried by this incident, but this source binds no field for "
                + ", ".join(unbound)
                + " — an absent or stale entity binding on the source, not a missing entity"
            )
    return "no declared identity_keys candidate is fully satisfied by this incident's entities"


# --- a literal that is only part of the stored value -------------------------------------
# `ValueForm.match` declares which position the value occupies in a composite identifier.
# Offers the pattern beside the equality. Confined to `field_mapping.match_patterns`; refuses
# negations.

# Canonical wildcard spelling -> each dialect's own. `?` is one character, `*` is any run.
_MATCH_WILDCARDS = {
    "sql": {"?": "_", "*": "%"},
    # `ES|QL`'s `LIKE` uses the canonical spelling already, as does the Query DSL `wildcard`.
    "esql": {"?": "?", "*": "*"},
}


def _match_pattern_in(value: str, pattern: str, dialect: str) -> Optional[str]:
    """``pattern`` in ``dialect``'s wildcard syntax, or ``None`` to decline.

    Declines when the value carries a character the target dialect reads as a wildcard: the
    pattern arrives with the value already substituted, so translating it would turn one
    character of the subject's own identifier into "any character". The shapes this exists for
    are alphanumeric identifier segments.
    """
    text = str(value)
    forbidden = "_%" if dialect != "esql" else "?*"
    if any(ch in text for ch in forbidden) or "'" in text or '"' in text:
        return None
    table = _MATCH_WILDCARDS.get(dialect, _MATCH_WILDCARDS["sql"])
    return "".join(table.get(ch, ch) for ch in str(pattern))


def _match_literal(pattern: str, dialect: str) -> str:
    """One dialect's string literal for a wildcard pattern."""
    if dialect == "esql":
        return '"' + str(pattern).replace("\\", "\\\\").replace('"', '\\"') + '"'
    return _quoted(pattern)


def widen_match_patterns(
    text: str,
    pattern_values: Dict[str, Dict[str, str]],
    source_name: str = "?",
    dialect: str = "sql",
) -> str:
    """Offer a value's declared match pattern beside the equality, on the same column.

    ``pattern_values`` is ``{real column: {value: pattern}}``, from ``field_mapping.match_patterns``.
    Two shapes: ``col = 'v'`` and ``col IN ('v', ...)`` become that predicate OR a ``LIKE`` per
    patterned value, parenthesised. Negations declined. Returns ``text`` unchanged when there
    is nothing to widen.
    """
    if not text or not pattern_values:
        return text
    out = text
    widened: List[str] = []
    declined: List[str] = []
    for column, pairs in pattern_values.items():
        usable: Dict[str, str] = {}
        for value, pattern in (pairs or {}).items():
            if not value or not pattern:
                continue
            rendered = _match_pattern_in(str(value), str(pattern), dialect)
            if rendered is None:
                declined.append(
                    f"{column}: {value!r} carries a character this backend reads as a "
                    "wildcard, so its declared pattern is not rendered"
                )
                continue
            usable[str(value)] = rendered
        if not usable:
            continue
        col_re = _stem_column_pattern(column)
        like_for = {
            value: f"{leaf_of(column)} LIKE {_match_literal(p, dialect)}"
            for value, p in usable.items()
        }

        # The list first: `render_filters` emits `IN` for every multi-value column. `NOT` is
        # matched on both sides of the column so a negation is declined explicitly rather than
        # failing to match by accident.
        list_re = re.compile(
            r"(?P<not>\bNOT\s+)?"
            + col_re
            + r"\s*(?P<not2>\bNOT\s+)?\bIN\s*\((?P<body>[^()]*)\)",
            re.IGNORECASE,
        )

        def _list(m: "re.Match", column=column, usable=usable, like_for=like_for) -> str:
            if m.group("not") or m.group("not2"):
                declined.append(
                    f"{column}: a NOT IN list — widening it would narrow the result"
                )
                return m.group(0)
            body = m.group("body")
            additions = [
                like_for[value]
                for value in usable
                if _quoted(value) in body or f'"{value}"' in body
            ]
            additions = [a for a in additions if a not in out]
            if not additions:
                return m.group(0)
            widened.append(f"{column} IN (…) OR {' OR '.join(additions)}")
            return "(" + m.group(0) + " OR " + " OR ".join(additions) + ")"

        out = list_re.sub(_list, out)

        # Then the equality. `==` is accepted because one dialect here spells it that way;
        # nothing else is, so `!=` / `<>` / an existing `LIKE` fall through untouched by
        # construction rather than by an exclusion list that could go stale.
        eq_re = re.compile(
            r"(?P<not>\bNOT\s+)?"
            + col_re
            + r"\s*(?P<op>==|=)\s*(?P<lit>'[^']*'|\"[^\"]*\")",
            re.IGNORECASE,
        )

        def _eq(m: "re.Match", column=column, usable=usable, like_for=like_for) -> str:
            literal = m.group("lit")[1:-1]
            if literal not in usable:
                return m.group(0)
            if m.group("not"):
                declined.append(
                    f"{column}: a negated equality — widening it would narrow the result"
                )
                return m.group(0)
            clause = like_for[literal]
            if clause in out:
                # Already offered in this query (the generator followed the hint, or the list
                # rewrite above did it). Publishing the same fact twice reads to an operator
                # as two predicates.
                return m.group(0)
            widened.append(f"{column} = {m.group('lit')} OR {clause}")
            return "(" + m.group(0) + " OR " + clause + ")"

        out = eq_re.sub(_eq, out)

    for note in widened:
        logger.warning(
            "Source '%s': widened a predicate to the declared match pattern of its value — "
            "%s. The incident carries only a fixed-width PART of this identifier, so an "
            "equality against it matches no row of a column storing the whole value, while "
            "reporting 0 rows as a success.",
            source_name,
            note,
        )
    for note in declined:
        logger.info(
            "Source '%s': left a predicate as generated rather than widening it to a match "
            "pattern — %s.",
            source_name,
            note,
        )
    return out


def widen_match_patterns_dsl(
    dsl: Any, pattern_values: Dict[str, Dict[str, str]], source_name: str = "?"
) -> Any:
    """DSL twin of :func:`widen_match_patterns`.

    ``term`` / ``terms`` becomes a ``bool.should`` holding the original clause beside one
    ``wildcard`` per patterned value, with an explicit ``minimum_should_match: 1`` (the default
    is 1 only while the ``bool`` carries no ``must``, and this clause may be spliced under one).
    That is the same widening shape ``key_was_enforced_dsl`` reads as conjunctive: every arm
    constrains the one column, and one arm is its equality.

    Declines on anything else: a clause naming two fields, a ``term`` whose value is an object,
    a value with no declared pattern. Skips everything under ``must_not``: widening a negation
    narrows. The input is not mutated.
    """
    if not isinstance(dsl, (dict, list)) or not pattern_values:
        return dsl
    by_leaf: Dict[str, Dict[str, str]] = {}
    for column, pairs in pattern_values.items():
        usable = {
            str(v): str(p) for v, p in (pairs or {}).items() if v and p and str(v) != str(p)
        }
        if usable:
            by_leaf[leaf_of(column)] = usable
    if not by_leaf:
        return dsl
    widened: List[str] = []

    def _widen_clause(kind: str, body: Dict) -> Optional[Dict]:
        field, spec = next(iter(body.items()))
        usable = by_leaf.get(leaf_of(str(field)))
        if not usable:
            return None
        if kind == "term":
            if not isinstance(spec, (str, int)):
                return None
            patterns = [usable[str(spec)]] if str(spec) in usable else []
        else:
            if not isinstance(spec, list):
                return None
            patterns = [usable[str(v)] for v in spec if str(v) in usable]
        patterns = list(dict.fromkeys(patterns))
        if not patterns:
            return None
        should: List[Dict[str, Any]] = [{kind: copy.deepcopy(body)}]
        should += [{"wildcard": {field: {"value": p}}} for p in patterns]
        widened.append(f"{field}: {kind} OR wildcard {', '.join(patterns)}")
        return {"bool": {"should": should, "minimum_should_match": 1}}

    def _walk(node: Any, negated: bool) -> Any:
        if isinstance(node, list):
            return [_walk(item, negated) for item in node]
        if not isinstance(node, dict):
            return node
        if not negated and len(node) == 1:
            key = str(next(iter(node)))
            body = node[key]
            if key in ("term", "terms") and isinstance(body, dict) and len(body) == 1:
                rebuilt = _widen_clause(key, body)
                if rebuilt is not None:
                    return rebuilt
        out: Dict[str, Any] = {}
        for key, value in node.items():
            # `must_not` negates everything under it at any depth: widening there narrows the
            # result, the one direction this guard must never take.
            out[key] = _walk(value, negated or str(key) == "must_not")
        return out

    result = _walk(copy.deepcopy(dsl), False)
    for note in widened:
        logger.warning(
            "Source '%s': widened a DSL clause to the declared match pattern of its value — "
            "%s. See the textual guard: the incident carries only a fixed-width PART of this "
            "identifier, and an equality against it matches no row of a column storing the "
            "whole value while reporting 0 rows as a success.",
            source_name,
            note,
        )
    return result
