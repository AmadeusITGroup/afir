"""Probe a live source before declaring anything; one library, every backend.

Every question uses the same retriever the run builds. Three invariants: failures are
explicit (``Measurement.ok`` beside ``rows``), a control is part of every comparative
measurement (``Comparison.NOT_TESTED`` on zero-vs-zero), and the write path is absent
structurally (:func:`_read_only_reason` checks first token and semicolons).

CLI: ``python -m src.knowledge.pack_probe <cmd>`` where ``<cmd>`` is one of
``sources``, ``leaves``, ``population``, ``values``, ``selectivity``, ``spread``,
``pair``, ``sql``. Tests: ``tests/test_pack_probe.py``.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Rows to pull when the question is "what does a row of this look like".
SAMPLE_ROWS = 5

#: Buckets returned by a grouped value or length question, unless the caller says otherwise.
TOP_VALUES = 20

#: The first token every readable statement may start with. A probe that can only read is a
#: property of this tuple, so a dialect gaining a new read verb is a review of one line.
_READ_VERBS = ("select", "with", "show", "describe", "desc", "explain", "from", "row")


def _read_only_reason(text: str) -> str:
    """Why ``text`` must not be sent, or ``""`` when it may be.

    Two checks. A leading-verb test alone is satisfied by ``SELECT 1; DROP TABLE t``,
    so a ``;`` separating two non-empty statements is refused outright (trailing
    semicolons are allowed). Refused rather than sanitised: a rewritten statement is one
    whose meaning nobody checked.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "the statement is empty"
    body = "\n".join(
        line for line in stripped.splitlines() if not line.strip().startswith("--")
    ).strip()
    if not body:
        return "the statement is only comments"
    parts = [p for p in body.split(";") if p.strip()]
    if len(parts) > 1:
        return (
            f"the statement chains {len(parts)} statements with ';' — a probe sends one "
            "read at a time, so this is refused rather than split"
        )
    first = body.lstrip("( \t\n").split(None, 1)
    verb = (first[0] if first else "").lower()
    if verb not in _READ_VERBS:
        return (
            f"the statement starts with {verb!r}, which is not one of the read verbs "
            f"({', '.join(_READ_VERBS)}) — this module has no write path"
        )
    return ""


def load_main_config() -> Dict:
    """The run's own config, env references expanded, without importing the pipeline eagerly.

    ``load_config`` lives in the pipeline entrypoint, which is a flat-import module, so the
    repo's ``src`` directory goes on ``sys.path`` first — the same thing ``pytest.ini`` does
    and the same thing every script under ``scripts/`` does. Reimplementing the expansion
    here instead would be a second answer to "what does ``${VAR}`` mean", and the copy that
    drifts is the one that hands a retriever the literal text ``${...}`` as a password — an
    auth failure that reads like a source with nothing to say.
    """
    import sys

    from src.utils.paths import REPO_ROOT, config_path

    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from src.main import load_config  # noqa: E402 — after the path is set up

    return load_config(config_path("main_config.yaml"))


# --------------------------------------------------------------------------- results


@dataclass
class Measurement:
    """One question asked of one source, and what came back — INCLUDING nothing.

    ``ok`` is the whole point. ``rows == []`` with ``ok=True`` is an answer (the source
    ran the query and matched nothing); ``rows == []`` with ``ok=False`` is a non-answer,
    and the two license opposite conclusions. Every helper in this module returns one of
    these, and none of them collapses the second into the first.
    """

    question: str
    source: str
    ok: bool
    rows: List[Dict] = field(default_factory=list)
    error: str = ""
    query: Any = ""
    elapsed_s: float = 0.0
    truncated: bool = False

    @property
    def n(self) -> int:
        """The first numeric cell of the first row, or the row count.

        A counting question answers with one row of one number; a listing question answers
        with rows. Both are read here so a caller can compare them without knowing which
        shape the dialect chose.
        """
        if self.rows and len(self.rows) == 1:
            for value in self.rows[0].values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    return int(value)
        return len(self.rows)

    def summary(self) -> str:
        if not self.ok:
            return f"FAILED ({self.error}) — not an empty result, a non-answer"
        cap = ", AND THERE WERE MORE (row cap)" if self.truncated else ""
        return f"{self.n} in {self.elapsed_s:.1f}s{cap}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"[{self.source}] {self.question}: {self.summary()}"


@dataclass
class Comparison:
    """A measurement beside the control that makes it mean something.

    ``subject`` asked the interesting question; ``control`` asked the same question with
    the interesting part removed. Four outcomes, and three of them are routinely reported
    as the fourth by a probe that only ran the subject.
    """

    subject: Measurement
    control: Measurement
    what: str = ""

    @property
    def rate(self) -> Optional[float]:
        if not (self.subject.ok and self.control.ok) or self.control.n <= 0:
            return None
        return self.subject.n / self.control.n

    def verdict(self) -> str:
        """One line stating which of the four outcomes this is."""
        if not self.control.ok:
            return (
                f"NOT TESTED — the control failed ({self.control.error}); nothing is "
                "known about the subject either way"
            )
        if not self.subject.ok:
            return (
                f"FAILED — the subject query did not run ({self.subject.error}), while "
                f"the control returned {self.control.n}. This is not an empty result"
            )
        if self.control.n == 0:
            return (
                "NOT TESTED — the control returned 0 too, so the subject was never "
                "exercised. Widen the control (drop a filter, widen the window) and "
                "re-ask; a zero here says nothing about the subject"
            )
        rate = self.rate or 0.0
        if self.subject.n == 0:
            return (
                f"EMPTY, and tested — 0 of {self.control.n} rows matched. The control ran, "
                "so this is a finding about the subject and not about the source"
            )
        return (
            f"{self.subject.n} of {self.control.n} rows ({rate:.2%})"
            + (", AND THE SUBJECT HIT THE ROW CAP" if self.subject.truncated else "")
        )

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"[{self.subject.source}] {self.what or self.subject.question}: {self.verdict()}"


class Unsupported(Exception):
    """This question has no faithful expression in this backend's language.

    Raised rather than approximated. An approximation is the defect this whole module is
    about: a number that looks like the measurement asked for and answers a different
    question is worse than no number, because it gets written into a declaration.
    """


# -------------------------------------------------------------------------- dialects
# A dialect writes a question and reads the answer but never executes; the Probe owns
# retriever dispatch. Predicates are opaque (str for SQL/ES-QL, dict for Query DSL),
# which lets the same seven helpers serve all three backends.


@dataclass
class _Request:
    """A built question: what to send, and how to turn the raw answer into rows."""

    payload: Any
    parse: Callable[[Any], List[Dict]]
    kind: str = "statement"


def _rows_as_is(raw: Any) -> List[Dict]:
    """Normalise a row list to lower-case keys.

    One backend upper-cases every identifier it returns, so a caller reading ``row["n"]``
    reads ``None`` on exactly one of the two SQL backends — the kind of difference that
    shows up as a wrong measurement rather than an error.
    """
    out = []
    for row in raw or []:
        if isinstance(row, dict):
            out.append({str(k).lower(): v for k, v in row.items()})
        else:
            out.append({"value": row})
    return out


class _Sql:
    """ANSI-ish SQL, spoken by the warehouse backends."""

    name = "sql"

    def and_(self, *predicates: Optional[str]) -> Optional[str]:
        parts = [p for p in predicates if p]
        return " AND ".join(f"({p})" for p in parts) if parts else None

    def not_null(self, column: str) -> str:
        return f"{column} IS NOT NULL"

    def not_blank(self, column: str) -> str:
        # Separate from NOT NULL: a column of empty strings looks fully populated to a
        # null check; a pack that declares it inhabited binds nothing.
        return f"{column} IS NOT NULL AND TRIM(CAST({column} AS STRING)) <> ''"

    def _where(self, predicate: Optional[str]) -> str:
        return f" WHERE {predicate}" if predicate else ""

    def sample(self, table: str, limit: int) -> _Request:
        return _Request(f"SELECT * FROM {table} LIMIT {int(limit)}", _rows_as_is)

    def count(self, table: str, predicate: Optional[str] = None) -> _Request:
        return _Request(
            f"SELECT COUNT(*) AS n FROM {table}{self._where(predicate)}", _rows_as_is
        )

    def distinct_count(
        self, table: str, column: str, predicate: Optional[str] = None
    ) -> _Request:
        return _Request(
            f"SELECT COUNT(DISTINCT {column}) AS n FROM {table}"
            f"{self._where(predicate)}",
            _rows_as_is,
        )

    def values(
        self, table: str, column: str, top: int, predicate: Optional[str] = None
    ) -> _Request:
        return _Request(
            f"SELECT {column} AS value, COUNT(*) AS n FROM {table}"
            f"{self._where(predicate)} GROUP BY {column} ORDER BY n DESC "
            f"LIMIT {int(top)}",
            _rows_as_is,
        )

    def length_spread(
        self, table: str, column: str, top: int, predicate: Optional[str] = None
    ) -> _Request:
        return _Request(
            f"SELECT LENGTH(CAST({column} AS STRING)) AS value, COUNT(*) AS n "
            f"FROM {table}{self._where(predicate)} "
            f"GROUP BY LENGTH(CAST({column} AS STRING)) ORDER BY n DESC "
            f"LIMIT {int(top)}",
            _rows_as_is,
        )


class _Esql:
    """The piped query language of one of the search backends.

    Close enough to SQL to share the Probe's helpers and far enough to need its own
    builder: there is no ``WHERE`` keyword after a ``FROM``, aggregation is ``STATS``, and
    the limit is a pipe stage.
    """

    name = "esql"

    def and_(self, *predicates: Optional[str]) -> Optional[str]:
        parts = [p for p in predicates if p]
        return " AND ".join(f"({p})" for p in parts) if parts else None

    def not_null(self, column: str) -> str:
        return f"{column} IS NOT NULL"

    def not_blank(self, column: str) -> str:
        return f'{column} IS NOT NULL AND {column} != ""'

    def _pipe(self, table: str, predicate: Optional[str]) -> str:
        head = f"FROM {table}"
        return f"{head} | WHERE {predicate}" if predicate else head

    def sample(self, table: str, limit: int) -> _Request:
        return _Request(f"FROM {table} | LIMIT {int(limit)}", _rows_as_is)

    def count(self, table: str, predicate: Optional[str] = None) -> _Request:
        return _Request(
            f"{self._pipe(table, predicate)} | STATS n = COUNT(*)", _rows_as_is
        )

    def distinct_count(
        self, table: str, column: str, predicate: Optional[str] = None
    ) -> _Request:
        return _Request(
            f"{self._pipe(table, predicate)} | STATS n = COUNT_DISTINCT({column})",
            _rows_as_is,
        )

    def values(
        self, table: str, column: str, top: int, predicate: Optional[str] = None
    ) -> _Request:
        return _Request(
            f"{self._pipe(table, predicate)} | STATS n = COUNT(*) BY value = {column} "
            f"| SORT n DESC | LIMIT {int(top)}",
            _rows_as_is,
        )

    def length_spread(
        self, table: str, column: str, top: int, predicate: Optional[str] = None
    ) -> _Request:
        return _Request(
            f"{self._pipe(table, predicate)} | EVAL value = LENGTH({column}) "
            f"| STATS n = COUNT(*) BY value | SORT n DESC | LIMIT {int(top)}",
            _rows_as_is,
        )


class _Dsl:
    """Query DSL over the search backend reached through its own gateway.

    Predicates here are DICTS — a query clause — and they compose with ``bool.filter``,
    which is why nothing outside this class ever looks inside one.
    """

    name = "dsl"

    def and_(self, *predicates: Optional[Dict]) -> Optional[Dict]:
        parts = [p for p in predicates if p]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return {"bool": {"filter": list(parts)}}

    def not_null(self, column: str) -> Dict:
        return {"exists": {"field": column}}

    def not_blank(self, column: str) -> Dict:
        # `exists` is true for an empty string, so the blank has to be excluded explicitly
        # — the same asymmetry the SQL dialect spells with TRIM.
        return {
            "bool": {
                "filter": [{"exists": {"field": column}}],
                "must_not": [{"term": {column: ""}}],
            }
        }

    def _query(self, predicate: Optional[Dict]) -> Dict:
        return predicate or {"match_all": {}}

    @staticmethod
    def _total(raw: Any) -> List[Dict]:
        total = ((raw or {}).get("hits") or {}).get("total")
        if isinstance(total, dict):
            # `relation: "gte"` means the backend stopped counting: a floor, not a count.
            # Reported as truncation by the Probe, which is the only honest reading.
            return [{"n": total.get("value", 0), "relation": total.get("relation", "eq")}]
        return [{"n": int(total or 0), "relation": "eq"}]

    @staticmethod
    def _hits(raw: Any) -> List[Dict]:
        hits = ((raw or {}).get("hits") or {}).get("hits") or []
        return [h.get("_source", h) if isinstance(h, dict) else {"value": h} for h in hits]

    @staticmethod
    def _buckets(raw: Any) -> List[Dict]:
        aggs = (raw or {}).get("aggregations") or {}
        buckets = ((aggs.get("probe") or {}).get("buckets")) or []
        return [
            {"value": b.get("key"), "n": b.get("doc_count", 0)}
            for b in buckets
            if isinstance(b, dict)
        ]

    @staticmethod
    def _cardinality(raw: Any) -> List[Dict]:
        aggs = (raw or {}).get("aggregations") or {}
        return [{"n": (aggs.get("probe") or {}).get("value", 0)}]

    def sample(self, table: str, limit: int) -> _Request:
        return _Request(
            {"size": int(limit), "query": {"match_all": {}}}, self._hits, kind="dsl"
        )

    def count(self, table: str, predicate: Optional[Dict] = None) -> _Request:
        return _Request(
            {"size": 0, "track_total_hits": True, "query": self._query(predicate)},
            self._total,
            kind="dsl",
        )

    def distinct_count(
        self, table: str, column: str, predicate: Optional[Dict] = None
    ) -> _Request:
        return _Request(
            {
                "size": 0,
                "query": self._query(predicate),
                "aggs": {"probe": {"cardinality": {"field": column}}},
            },
            self._cardinality,
            kind="dsl",
        )

    def values(
        self, table: str, column: str, top: int, predicate: Optional[Dict] = None
    ) -> _Request:
        return _Request(
            {
                "size": 0,
                "query": self._query(predicate),
                "aggs": {"probe": {"terms": {"field": column, "size": int(top)}}},
            },
            self._buckets,
            kind="dsl",
        )

    def length_spread(
        self, table: str, column: str, top: int, predicate: Optional[Dict] = None
    ) -> _Request:
        raise Unsupported(
            "a length histogram in Query DSL needs a runtime script, which this module "
            "will not send. Read the spread from `values` output instead, or ask the same "
            "question of a backend that can express it"
        )


#: Which language each retriever kind speaks and how the Probe reaches it. Attribute
#: names are the retrievers' raw-execution seams. Sources not listed have no composable
#: query language and are reported as unsupported rather than guessed at.
_DIALECTS: Dict[str, Tuple[Any, str]] = {
    "databricks": (_Sql(), "_execute_sql"),
    "snowflake": (_Sql(), "_execute_sql"),
    "elasticsearch": (_Esql(), "_execute_esql"),
    "kibana": (_Dsl(), "_search"),
}


# ----------------------------------------------------------------------------- probe


class Probe:
    """Ask measured questions of the sources a run would actually build.

    Built on :class:`LogRetrievalEngine`, so what is reachable here is exactly what is
    reachable there — including the ones that are NOT: ``unavailable()`` reports every
    declared source that built no retriever, which is the first thing to check when a
    probe "finds nothing" about a source (a skipped source cannot answer, and that is not
    a fact about the data).

    Constructed with **no LLM client**. Nothing here generates a query, so the code path
    that would is not merely unused, it cannot run — the same structural posture as the
    read-only statement guard.
    """

    def __init__(self, engine, pack=None, config: Optional[Dict] = None):
        self.engine = engine
        self.pack = pack
        self.config = config or {}

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    async def open(
        cls,
        *,
        pack_name: Optional[str] = None,
        config: Optional[Dict] = None,
        auth: Optional[Any] = None,
    ) -> "Probe":
        """Build the engine the way ``main()`` does, minus the pipeline.

        ``pack_name`` defaults to the configured one, because probing a source through a
        DIFFERENT pack's bindings would measure a source the run never asks that way.
        """
        from src.knowledge.pack import load_knowledge_pack
        from src.log_retrieval import LogRetrievalEngine
        from src.utils.paths import knowledge_pack_dir

        if config is None:
            config = load_main_config()
        name = pack_name or str(
            (config.get("knowledge", {}) or {}).get("pack_dir", "") or ""
        ).strip()
        pack = load_knowledge_pack(knowledge_pack_dir(name)) if name else None
        if auth is None:
            try:
                from src.utils.databricks_auth import try_build_auth

                auth = try_build_auth()
            except Exception as exc:  # noqa: BLE001 — a probe must run without the SDK
                logger.info("no unified auth available (%s); using static tokens", exc)
                auth = None
        engine = LogRetrievalEngine(
            config.get("log_sources", {}) or {},
            None,
            auth=auth,
            knowledge_pack=pack,
        )
        return cls(engine, pack=pack, config=config)

    async def close(self) -> None:
        for name, retriever in (self.engine.retrievers or {}).items():
            try:
                await retriever.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("closing %s raised: %s", name, exc)

    # -------------------------------------------------------------------- what exists

    def sources(self) -> Dict[str, str]:
        """Every source that built a retriever, mapped to the language it speaks."""
        out = {}
        for name, retriever in (self.engine.retrievers or {}).items():
            kind = self._kind(retriever)
            dialect = _DIALECTS.get(kind)
            out[name] = dialect[0].name if dialect else f"{kind} (no probe language)"
        return out

    def unavailable(self) -> Dict[str, str]:
        """Declared sources that built no retriever, and why. Read this FIRST."""
        return dict(self.engine.unavailable_sources or {})

    def tables(self, source: str) -> List[str]:
        """What the pack/config declared as this source's targets, if anything.

        A statement needs a target and the probe will not invent one. For the search
        backends the target is the configured index, which is why it can be answered here
        at all; for the warehouses it is the declared allow-list, and an empty list means
        the caller has to name the table (the source may reach a whole schema).
        """
        retriever = self._retriever(source)
        cfg = getattr(retriever, "config", {}) or {}
        declared = [t for t in (cfg.get("tables") or []) if t]
        if declared:
            return declared
        index = cfg.get("index")
        return [index] if index else []

    # ------------------------------------------------------------------- raw questions

    async def ask(self, source: str, request: _Request, question: str) -> Measurement:
        """Send one built request and record what happened — including that it did not.

        The single seam every helper goes through, for the same reason the retrievers have
        one ``publish_query``: a per-helper try/except is five copies of the decision about
        what an exception MEANS, and the copy that gets it wrong reports an empty result.
        """
        retriever = None
        try:
            retriever = self._retriever(source)
            kind = self._kind(retriever)
            dialect, seam = self._dialect(kind)
            if request.kind == "statement":
                reason = _read_only_reason(str(request.payload))
                if reason:
                    raise ValueError(f"refused: {reason}")
            call = getattr(retriever, seam, None)
            if call is None:
                raise AttributeError(
                    f"source '{source}' is a {kind} retriever with no '{seam}' seam; "
                    "the probe cannot reach it"
                )
            started = time.monotonic()
            raw = await call(request.payload)
            elapsed = time.monotonic() - started
            rows = request.parse(raw)
            cap = int(getattr(retriever, "max_results", 0) or 0)
            truncated = bool(cap) and len(rows) >= cap
            if rows and str(rows[0].get("relation", "eq")) != "eq":
                # A counted floor is a truncation of the COUNT rather than of the rows,
                # and it reads as an exact number to everything downstream.
                truncated = True
            logger.info(
                "probe %s: %s -> %d row(s) in %.1fs", source, question, len(rows), elapsed
            )
            return Measurement(
                question=question,
                source=source,
                ok=True,
                rows=rows,
                query=request.payload,
                elapsed_s=elapsed,
                truncated=truncated,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — the failure IS the result here
            # The TYPE, never just the text: a message is prose that varies per backend
            # and per release, and the one thing a reader needs is whether this was a
            # timeout, a missing column or a refusal.
            return Measurement(
                question=question,
                source=source,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                query=request.payload,
            )

    async def sql(self, source: str, statement: str, question: str = "") -> Measurement:
        """Send a statement the caller wrote, through the guard and the run's retriever."""
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        if dialect.name == "dsl":
            raise Unsupported(
                f"source '{source}' speaks Query DSL, not a statement language; use dsl()"
            )
        return await self.ask(
            source, _Request(statement, _rows_as_is), question or "an ad-hoc statement"
        )

    async def dsl(self, source: str, body: Dict, question: str = "") -> Measurement:
        """Send a Query DSL body the caller wrote (hits are returned as rows)."""
        return await self.ask(
            source, _Request(body, _Dsl._hits, kind="dsl"), question or "an ad-hoc body"
        )

    # ------------------------------------------------------------- the seven questions

    async def leaves(
        self, source: str, table: str, limit: int = SAMPLE_ROWS
    ) -> Measurement:
        """Which leaves a ROW of this table actually carries, flattened.

        Not the schema. A leaf on the schema doc is a leaf the projection MAY expose, and
        a condition reads the row: a documented column can be absent from every row a
        query returns (an alias, a view, a struct the projection dropped), and a field
        path that resolves to nothing degrades its condition to ``unknown`` while the
        retrieval reports success. So this samples and unions, and the row count is part
        of the answer — one row is one shape, and an index pattern or a partitioned table
        is a SET of shapes.
        """
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        request = dialect.sample(table, limit)
        m = await self.ask(source, request, f"a sample of {table}")
        if not m.ok:
            return m
        sampled = len(m.rows)
        paths: Dict[str, int] = {}
        for row in m.rows:
            for path in sorted(_flatten(row)):
                paths[path] = paths.get(path, 0) + 1
        m.rows = [
            {"path": p, "present_in_rows": c, "of_sampled": sampled}
            for p, c in sorted(paths.items())
        ]
        # The sampled count stays in the question, because a path present in 1 of 5 rows
        # and a path present in 5 of 5 are different declarations to write.
        m.question = f"the leaves present in {sampled} sampled row(s) of {table}"
        return m

    async def count(
        self, source: str, table: str, predicate: Any = None, question: str = ""
    ) -> Measurement:
        """How many rows — the cheapest question, and the one every other one needs."""
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        return await self.ask(
            source,
            dialect.count(table, predicate),
            question or f"rows in {table}" + (" matching a predicate" if predicate else ""),
        )

    async def population(
        self, source: str, table: str, column: str, predicate: Any = None
    ) -> Dict[str, Measurement]:
        """Is this column POPULATED — three numbers, because one of them lies.

        ``total`` / ``not_null`` / ``not_blank`` / ``distinct``. A column that is 100%
        non-null and 100% blank reads as fully populated to every check that asks one
        question, and a pack that binds an identity to it produces a valid predicate
        matching nothing. ``distinct`` is here because a column populated with ONE value
        is populated and useless — that is a constant, and a constant in an evidence group
        satisfies the group on its own.
        """
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        total = await self.count(source, table, predicate, question=f"rows in {table}")
        not_null = await self.count(
            source,
            table,
            dialect.and_(predicate, dialect.not_null(column)),
            question=f"{column} is not null",
        )
        not_blank = await self.count(
            source,
            table,
            dialect.and_(predicate, dialect.not_blank(column)),
            question=f"{column} is not null and not blank",
        )
        distinct = await self.ask(
            source,
            dialect.distinct_count(table, column, predicate),
            f"distinct values of {column}",
        )
        return {
            "total": total,
            "not_null": not_null,
            "not_blank": not_blank,
            "distinct": distinct,
        }

    async def values(
        self,
        source: str,
        table: str,
        column: str,
        top: int = TOP_VALUES,
        predicate: Any = None,
    ) -> Measurement:
        """The commonest values and their counts — the shape of what is stored.

        Where a notation is read rather than guessed: a masked value, a padded code, a
        prefix convention and a placeholder all announce themselves in the top buckets,
        and each of them changes what a predicate has to spell.
        """
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        return await self.ask(
            source,
            dialect.values(table, column, top, predicate),
            f"the top {top} values of {column} in {table}",
        )

    async def selectivity(
        self, source: str, table: str, predicate: Any, within: Any = None
    ) -> Comparison:
        """How much of the POPULATION this predicate matches — the base rate.

        The measurement that decides whether a value discriminates. A value that is on the
        subject AND on half the population is not evidence, however well it matches the
        wording of the report; the only way to know which it is, is to count both. The
        control is the same query with the predicate removed (optionally still inside
        ``within``, when the population of interest is a subset).
        """
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        subject = await self.count(
            source, table, dialect.and_(within, predicate), question="the predicate"
        )
        control = await self.count(
            source, table, within, question="the population it is measured against"
        )
        return Comparison(subject, control, what=f"selectivity of a predicate on {table}")

    async def spread(
        self,
        source: str,
        table: str,
        column: str,
        top: int = TOP_VALUES,
        predicate: Any = None,
    ) -> Measurement:
        """The LENGTHS stored in this column, most common first.

        A form question. One identifier may be stored at two precisions, or a column may
        hold a segment of the value a report names, and both are invisible to a
        value-equality test: the long form is a valid predicate matching nothing. A length
        histogram says which precision is in the column before a declaration commits to
        one. Raises :class:`Unsupported` on a backend whose language cannot express it,
        rather than approximating.
        """
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        return await self.ask(
            source,
            dialect.length_spread(table, column, top, predicate),
            f"the value lengths of {column} in {table}",
        )

    async def pair(
        self, source: str, table: str, left: Any, right: Any, within: Any = None
    ) -> Dict[str, Measurement]:
        """Do these two predicates hold of the SAME rows — the four cells, not one.

        ``both`` / ``left_only`` / ``right_only`` / ``population``. Two identifiers being
        the same actor, a key needing conjunction rather than disjunction, an exclusion
        that must not be read across sibling rows — every one of them is this shape, and
        every one of them is got wrong by measuring ``both`` alone. ``left_only`` > 0 with
        ``both`` == 0 is the answer that stops a merge.
        """
        dialect, _ = self._dialect(self._kind(self._retriever(source)))
        return {
            "both": await self.count(
                source, table, dialect.and_(within, left, right), question="both"
            ),
            "left": await self.count(
                source, table, dialect.and_(within, left), question="the left one"
            ),
            "right": await self.count(
                source, table, dialect.and_(within, right), question="the right one"
            ),
            "population": await self.count(source, table, within, question="all rows"),
        }

    async def cost(
        self, source: str, table: str, predicate: Any = None
    ) -> Measurement:
        """What this scan COSTS, measured against the budget it will run under.

        A slow query is not a slow query for long: it becomes a timeout, then an empty
        result, then INSUFFICIENT DATA over data that was there. The margin is the budget,
        so the elapsed time is reported beside the per-source timeout this source is
        actually configured with — a scan that finishes just inside it is a scan that fails
        on a busy day.
        """
        m = await self.count(source, table, predicate, question=f"a full scan of {table}")
        retriever = self._retriever(source)
        budget = float(
            (getattr(retriever, "config", {}) or {}).get("timeout", 0) or 0
        )
        if budget and m.ok:
            margin = budget - m.elapsed_s
            m.question += (
                f" (took {m.elapsed_s:.1f}s of a {budget:.0f}s budget; "
                f"{margin:.0f}s of margin)"
            )
        return m

    # ------------------------------------------------------------------------ internals

    def _retriever(self, source: str):
        retrievers = self.engine.retrievers or {}
        if source in retrievers:
            return retrievers[source]
        why = self.unavailable().get(source)
        if why:
            raise KeyError(
                f"source '{source}' is declared but built no retriever ({why}) — it "
                "cannot answer anything on this configuration, which is not a fact "
                "about the data"
            )
        raise KeyError(
            f"no source '{source}'. Built sources: "
            + (", ".join(sorted(retrievers)) or "(none)")
        )

    @staticmethod
    def _kind(retriever) -> str:
        cfg = getattr(retriever, "config", {}) or {}
        kind = str(cfg.get("type") or "").strip().lower()
        if kind:
            return kind
        # A retriever built from an explicit override may carry no `type`; fall back to
        # the class name, which every implementation spells after its backend.
        return type(retriever).__name__.replace("Retriever", "").lower()

    @staticmethod
    def _dialect(kind: str) -> Tuple[Any, str]:
        try:
            return _DIALECTS[kind]
        except KeyError:
            raise Unsupported(
                f"a '{kind}' source has no query language this module can compose "
                f"(it speaks: {', '.join(sorted(_DIALECTS))}). Ask it through the "
                "pipeline instead, or read its rows with the retriever's own seam"
            ) from None


def _flatten(value: Any, prefix: str = "") -> List[str]:
    """Every leaf path in a row, dotted, with a list rendered as ``[]``.

    A list is flattened through ONE synthetic segment rather than per index, because an
    array of structs is one shape repeated and a path carrying ``[3]`` is a path no
    declaration can use.
    """
    out: List[str] = []
    if isinstance(value, dict):
        if not value:
            return [prefix] if prefix else []
        for key, sub in value.items():
            out.extend(_flatten(sub, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{prefix}[]"] if prefix else []
        seen: List[str] = []
        for item in value[:5]:
            for path in _flatten(item, f"{prefix}[]"):
                if path not in seen:
                    seen.append(path)
        return seen
    return [prefix] if prefix else []


# --------------------------------------------------------------------- rendering + CLI


def render(obj: Any, *, max_rows: int = TOP_VALUES) -> str:
    """One measurement as text, for a terminal or a tool result.

    A cut row list SAYS it was cut. The row cap is the same class of bound as a retriever's
    ``max_results``, and this module's whole posture is that ``N rows`` and ``N rows, and
    there were more`` are different findings — so the caller reading twenty buckets of a
    hundred has to be told which of the two it is holding.
    """
    if isinstance(obj, Measurement):
        lines = [str(obj)]
        shown = obj.rows[: max(0, int(max_rows))]
        lines += [f"    {json.dumps(row, default=str)}" for row in shown]
        if len(obj.rows) > len(shown):
            lines.append(
                f"    ... {len(obj.rows) - len(shown)} further row(s) not shown "
                f"(this rendering caps at {max_rows})"
            )
        if obj.query:
            lines.append(f"    query: {obj.query}")
        return "\n".join(lines)
    if isinstance(obj, Comparison):
        return str(obj)
    if isinstance(obj, dict):
        out = []
        for key, value in obj.items():
            if isinstance(value, Measurement):
                out.append(f"  {key}: {value.summary()}")
            elif isinstance(value, Comparison):
                out.append(f"  {key}: {value.verdict()}")
            else:
                out.append(f"  {key}: {value}")
        return "\n".join(out)
    return str(obj)


def _print(obj: Any) -> None:
    print(render(obj))


async def _run(args) -> int:
    probe = await Probe.open(pack_name=args.pack)
    try:
        unavailable = probe.unavailable()
        if unavailable and args.command != "sources":
            print(f"NOTE: {len(unavailable)} declared source(s) built no retriever "
                  f"and cannot answer: {', '.join(sorted(unavailable))}")
        if args.command == "sources":
            for name, language in sorted(probe.sources().items()):
                print(f"  {name}: {language}  targets={probe.tables(name) or '(caller must name one)'}")
            for name, why in sorted(unavailable.items()):
                print(f"  {name}: UNAVAILABLE — {why}")
            return 0
        if args.command == "leaves":
            _print(await probe.leaves(args.source, args.table))
        elif args.command == "count":
            _print(await probe.count(args.source, args.table, args.where))
        elif args.command == "population":
            _print(await probe.population(args.source, args.table, args.column, args.where))
        elif args.command == "values":
            _print(await probe.values(args.source, args.table, args.column, args.top, args.where))
        elif args.command == "selectivity":
            _print(await probe.selectivity(args.source, args.table, args.predicate, args.where))
        elif args.command == "spread":
            _print(await probe.spread(args.source, args.table, args.column, args.top, args.where))
        elif args.command == "pair":
            _print(await probe.pair(args.source, args.table, args.left, args.right, args.where))
        elif args.command == "cost":
            _print(await probe.cost(args.source, args.table, args.where))
        elif args.command == "sql":
            _print(await probe.sql(args.source, args.statement))
        elif args.command == "dsl":
            _print(await probe.dsl(args.source, json.loads(args.body)))
        return 0
    finally:
        await probe.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.knowledge.pack_probe",
        description=(
            "Measure a source before declaring anything about it. Read-only: every "
            "statement is checked for a read verb and refused if it chains."
        ),
    )
    parser.add_argument("--pack", default=None, help="pack directory name (default: configured)")
    sub = parser.add_subparsers(dest="command", required=True)

    def target(p, *, column=False, where=True, top=False):
        p.add_argument("source")
        p.add_argument("table")
        if column:
            p.add_argument("column")
        if top:
            p.add_argument("--top", type=int, default=TOP_VALUES)
        if where:
            p.add_argument("--where", default=None, help="a predicate in the source's own language")
        return p

    sub.add_parser("sources", help="what is reachable, in which language, and what is not")
    target(sub.add_parser("leaves", help="the leaves a sampled ROW carries"), where=False)
    target(sub.add_parser("count", help="how many rows"))
    target(sub.add_parser("population", help="total / not-null / not-blank / distinct"), column=True)
    target(sub.add_parser("values", help="the commonest values"), column=True, top=True)
    target(sub.add_parser("spread", help="the value LENGTHS stored"), column=True, top=True)
    target(sub.add_parser("cost", help="elapsed against the configured budget"))

    sel = target(sub.add_parser("selectivity", help="the base rate of a predicate"))
    sel.add_argument("predicate")
    pair = target(sub.add_parser("pair", help="do two predicates hold of the same rows"))
    pair.add_argument("left")
    pair.add_argument("right")

    adhoc = sub.add_parser("sql", help="one read statement, guarded")
    adhoc.add_argument("source")
    adhoc.add_argument("statement")
    body = sub.add_parser("dsl", help="one Query DSL body (JSON)")
    body.add_argument("source")
    body.add_argument("body")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
