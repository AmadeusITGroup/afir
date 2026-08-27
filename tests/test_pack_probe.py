"""The probe library: it asks through the run's own retriever, and it never loses a failure.

WHAT THIS FILE IS GUARDING, and why each one is invisible otherwise:

* **A failure reported as an empty result.** The defect the whole module exists to prevent,
  and the natural shape of every hand-rolled probe (`except Exception: return []`). Asserted
  from both ends: a raising seam must come back `ok=False` with the exception TYPE, and a
  comparison whose control failed must refuse to call the subject a finding.
* **A zero-vs-zero result read as a finding.** "0 rows matched" means something only if the
  control returned rows. `Comparison.verdict()` has four branches and three of them are
  routinely reported as the fourth.
* **A write reaching a system of record from an authoring aid.** The guard is a property of
  one tuple and one `;` check, so it is asserted by case — including the chained statement,
  which a leading-verb check alone accepts.
* **A dialect that silently answers a different question.** Each backend's builder is asked
  for the same seven measurements and the emitted text/body is asserted, because a builder
  that drops the predicate returns a plausible number for the wrong population. Where a
  dialect cannot express one faithfully it must RAISE, not approximate.
* **A source that cannot answer, read as a source with nothing to say.** A declared source
  that built no retriever has to be distinguishable from an empty one, by message.

No backend and no LLM is touched: the retrievers here are fakes recording what they were
handed, which is exactly the seam the real ones expose.
"""

import asyncio

import pytest

from src.knowledge import pack_probe
from src.knowledge.pack_probe import (Comparison, Measurement, Probe, Unsupported,
                                      _read_only_reason)


class FakeSql:
    """A warehouse-shaped retriever: one `_execute_sql` seam, statements recorded."""

    def __init__(self, rows=None, raises=None, kind="databricks", max_results=500):
        self.config = {"type": kind, "name": "s", "tables": ["t_one", "t_two"]}
        self.max_results = max_results
        self.rows = rows if rows is not None else [{"N": 7}]
        self.raises = raises
        self.seen = []

    async def _execute_sql(self, statement):
        self.seen.append(statement)
        if self.raises:
            raise self.raises
        return self.rows(statement) if callable(self.rows) else self.rows

    async def close(self):
        pass


class FakeDsl:
    """A gateway-shaped retriever: one `_search` seam over Query DSL bodies."""

    def __init__(self, raw=None, raises=None):
        self.config = {"type": "kibana", "name": "d", "index": "idx-*"}
        self.max_results = 500
        self.raw = raw if raw is not None else {"hits": {"total": {"value": 3, "relation": "eq"}}}
        self.raises = raises
        self.seen = []

    async def _search(self, body):
        self.seen.append(body)
        if self.raises:
            raise self.raises
        return self.raw(body) if callable(self.raw) else self.raw

    async def close(self):
        pass


class FakeEngine:
    def __init__(self, retrievers, unavailable=None):
        self.retrievers = retrievers
        self.unavailable_sources = unavailable or {}


def probe(**retrievers):
    unavailable = retrievers.pop("_unavailable", None)
    return Probe(FakeEngine(retrievers, unavailable))


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------- the read-only guard


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT COUNT(*) FROM t",
        "select 1",
        "  WITH x AS (SELECT 1) SELECT * FROM x",
        "SHOW TABLES",
        "DESCRIBE t",
        "-- a comment\nSELECT 1",
        "SELECT 1;",
        "FROM idx | STATS n = COUNT(*)",
    ],
)
def test_a_read_is_allowed(statement):
    assert _read_only_reason(statement) == ""


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM t",
        "DROP TABLE t",
        "UPDATE t SET a = 1",
        "INSERT INTO t VALUES (1)",
        "MERGE INTO t USING s ON a = b",
        "TRUNCATE TABLE t",
        "CREATE TABLE t AS SELECT 1",
        "GRANT ALL ON t TO x",
        "",
        "   ",
        "-- only a comment",
    ],
)
def test_a_write_is_refused_by_verb(statement):
    assert _read_only_reason(statement)


def test_a_chained_statement_is_refused_even_though_it_starts_with_a_read():
    """The case a leading-verb check accepts, and the one a built predicate produces.

    `SELECT 1; DROP TABLE t` passes every first-token test there is. It is refused rather
    than split, because a probe sends one read at a time and a statement somebody rewrote
    is a statement nobody checked.
    """
    reason = _read_only_reason("SELECT 1; DROP TABLE t")
    assert "chains" in reason


def test_the_guard_runs_before_the_seam_is_called():
    """Refused means NOT SENT — asserted on the fake, not on the return value."""
    r = FakeSql()
    m = run(probe(s=r).sql("s", "DELETE FROM t"))
    assert m.ok is False
    assert "refused" in m.error
    assert r.seen == [], "the statement reached the backend"


def test_a_dsl_body_is_not_run_through_the_statement_guard():
    """A JSON body has no verb; screening it as text would refuse every DSL probe."""
    r = FakeDsl(raw={"hits": {"hits": [{"_source": {"a": 1}}]}})
    m = run(probe(d=r).dsl("d", {"query": {"match_all": {}}}))
    assert m.ok and m.rows == [{"a": 1}]


# ------------------------------------------------------------- failure is not emptiness


def test_a_raising_seam_is_a_non_answer_and_keeps_the_exception_type():
    m = run(probe(s=FakeSql(raises=TimeoutError("gave up"))).count("s", "t"))
    assert m.ok is False
    assert m.rows == []
    assert "TimeoutError" in m.error
    assert "FAILED" in m.summary() and "non-answer" in m.summary()


def test_a_clean_empty_result_is_an_answer():
    m = run(probe(s=FakeSql(rows=[])).count("s", "t"))
    assert m.ok is True
    assert m.n == 0
    assert "FAILED" not in m.summary()


def test_the_two_are_distinguishable_from_the_outside():
    """The one property everything downstream rests on, stated as one assertion."""
    failed = run(probe(s=FakeSql(raises=ValueError("x"))).count("s", "t"))
    empty = run(probe(s=FakeSql(rows=[])).count("s", "t"))
    assert failed.rows == empty.rows == []
    assert failed.ok != empty.ok


def test_a_hit_row_cap_is_reported_as_a_floor():
    r = FakeSql(rows=[{"a": i} for i in range(5)], max_results=5)
    m = run(probe(s=r).sql("s", "SELECT * FROM t"))
    assert m.truncated
    assert "THERE WERE MORE" in m.summary()


def test_a_counted_floor_from_the_search_backend_is_truncation_too():
    """`relation: gte` is a number that reads exactly like a count and is not one."""
    r = FakeDsl(raw={"hits": {"total": {"value": 10000, "relation": "gte"}}})
    m = run(probe(d=r).count("d", "idx-*"))
    assert m.ok and m.n == 10000 and m.truncated


# ------------------------------------------------------------------------- comparison


def _m(ok=True, n=0):
    return Measurement(question="q", source="s", ok=ok, rows=[{"n": n}] if ok else [])


def test_zero_against_a_zero_control_is_not_tested():
    verdict = Comparison(_m(n=0), _m(n=0)).verdict()
    assert "NOT TESTED" in verdict


def test_zero_against_a_populated_control_is_a_finding():
    verdict = Comparison(_m(n=0), _m(n=1200)).verdict()
    assert "EMPTY, and tested" in verdict
    assert "1200" in verdict


def test_a_failed_control_reports_nothing_known_either_way():
    c = Comparison(_m(n=0), Measurement(question="q", source="s", ok=False, error="Boom: x"))
    assert "NOT TESTED" in c.verdict()
    assert c.rate is None


def test_a_failed_subject_is_not_reported_as_an_empty_one():
    c = Comparison(
        Measurement(question="q", source="s", ok=False, error="TimeoutError: t"), _m(n=90)
    )
    assert "FAILED" in c.verdict()
    assert "not an empty result" in c.verdict()


def test_a_base_rate_is_reported_as_a_share_of_the_population():
    c = Comparison(_m(n=340), _m(n=586))
    assert "340 of 586" in c.verdict()
    assert c.rate == pytest.approx(340 / 586)


# ------------------------------------------------------------- the seven measurements


def test_selectivity_asks_twice_and_the_control_drops_the_predicate():
    """The control is the same question minus the interesting part — asserted by text."""
    r = FakeSql(rows=lambda s: [{"n": 3}] if "col = 'x'" in s else [{"n": 100}])
    c = run(probe(s=r).selectivity("s", "t", "col = 'x'"))
    assert len(r.seen) == 2
    assert "col = 'x'" in r.seen[0]
    assert "col = 'x'" not in r.seen[1]
    assert "3 of 100" in c.verdict()


def test_selectivity_keeps_the_population_bound_on_both_sides():
    """`within` narrows the population; dropping it from the control inflates the rate."""
    r = FakeSql()
    run(probe(s=r).selectivity("s", "t", "col = 'x'", within="d > 0"))
    assert all("d > 0" in seen for seen in r.seen)


def test_population_asks_the_three_numbers_that_can_disagree():
    r = FakeSql()
    out = run(probe(s=r).population("s", "t", "col"))
    assert set(out) == {"total", "not_null", "not_blank", "distinct"}
    assert "IS NOT NULL" in r.seen[1]
    assert "TRIM" in r.seen[2], "a blank-string column reads as populated without this"
    assert "COUNT(DISTINCT col)" in r.seen[3]


def test_pair_asks_all_four_cells():
    r = FakeSql()
    out = run(probe(s=r).pair("s", "t", "a = 1", "b = 2"))
    assert set(out) == {"both", "left", "right", "population"}
    assert "a = 1" in r.seen[0] and "b = 2" in r.seen[0]
    assert "b = 2" not in r.seen[1]
    assert "a = 1" not in r.seen[2]
    assert "WHERE" not in r.seen[3]


def test_values_groups_and_orders_by_frequency():
    r = FakeSql(rows=[{"value": "a", "n": 9}])
    m = run(probe(s=r).values("s", "t", "col", top=3))
    assert "GROUP BY col" in r.seen[0] and "LIMIT 3" in r.seen[0]
    assert m.rows == [{"value": "a", "n": 9}]


def test_leaves_reads_the_ROW_and_not_the_schema():
    """A leaf on the schema is not a leaf in the row; arrays flatten through one segment."""
    r = FakeSql(
        rows=[
            {"a": 1, "b": {"c": 2}, "d": [{"e": 3}]},
            {"a": 1},
        ]
    )
    m = run(probe(s=r).leaves("s", "t"))
    paths = {row["path"]: row["present_in_rows"] for row in m.rows}
    assert paths == {"a": 2, "b.c": 1, "d[].e": 1}
    assert "2 sampled row(s)" in m.question


def test_cost_states_the_margin_against_the_configured_budget():
    r = FakeSql()
    r.config["timeout"] = 300
    m = run(probe(s=r).cost("s", "t"))
    assert "budget" in m.question and "margin" in m.question


def test_a_number_is_read_whatever_case_the_backend_returns_it_in():
    """One backend upper-cases every identifier; a caller reading `row["n"]` gets None."""
    m = run(probe(s=FakeSql(rows=[{"N": 42}])).count("s", "t"))
    assert m.rows == [{"n": 42}]
    assert m.n == 42


# ------------------------------------------------------------------------- dialects


def test_the_search_backend_speaks_its_own_piped_language():
    esql = pack_probe._Esql()
    assert esql.count("idx", "a == 1").payload == "FROM idx | WHERE a == 1 | STATS n = COUNT(*)"
    assert "COUNT_DISTINCT(col)" in esql.distinct_count("idx", "col").payload
    assert _read_only_reason(esql.values("idx", "col", 5).payload) == ""


def test_the_dsl_builders_compose_predicates_as_clauses_not_text():
    dsl = pack_probe._Dsl()
    both = dsl.and_({"term": {"a": 1}}, {"term": {"b": 2}})
    assert both == {"bool": {"filter": [{"term": {"a": 1}}, {"term": {"b": 2}}]}}
    assert dsl.and_(None, None) is None
    assert dsl.and_({"term": {"a": 1}}) == {"term": {"a": 1}}
    # `exists` is true of an empty string, so the blank has to be excluded explicitly.
    assert "must_not" in dsl.not_blank("col")["bool"]


def test_a_question_a_dialect_cannot_express_raises_instead_of_approximating():
    with pytest.raises(Unsupported):
        pack_probe._Dsl().length_spread("idx", "col", 5)
    with pytest.raises(Unsupported):
        run(probe(d=FakeDsl()).spread("d", "idx-*", "col"))


def test_a_backend_with_no_query_language_says_so_rather_than_guessing():
    class FakeRest:
        config = {"type": "rest", "name": "r"}

        async def close(self):
            pass

    with pytest.raises(Unsupported) as exc:
        run(probe(r=FakeRest()).count("r", "t"))
    assert "no query language" in str(exc.value)


def test_a_statement_backend_refuses_a_dsl_body_and_the_other_way_round():
    with pytest.raises(Unsupported):
        run(probe(d=FakeDsl()).sql("d", "SELECT 1"))


# ------------------------------------------------------------------- what is reachable


def test_the_sources_map_names_the_language_each_one_speaks():
    p = probe(w=FakeSql(), d=FakeDsl(), e=FakeSql(kind="elasticsearch"))
    assert p.sources() == {"w": "sql", "d": "dsl", "e": "esql"}


def test_targets_come_from_the_declaration_and_are_never_invented():
    p = probe(w=FakeSql(), d=FakeDsl())
    assert p.tables("w") == ["t_one", "t_two"]
    assert p.tables("d") == ["idx-*"]
    bare = FakeSql()
    bare.config = {"type": "databricks", "name": "b"}
    assert probe(b=bare).tables("b") == []


def test_a_declared_source_that_built_no_retriever_says_that_and_not_nothing():
    """A skipped source cannot answer, and that is not a fact about the data."""
    p = probe(_unavailable={"gone": "no credentials configured"})
    with pytest.raises(KeyError) as exc:
        p.tables("gone")
    assert "no credentials configured" in str(exc.value)
    assert "not a fact about the data" in str(exc.value)


def test_an_unknown_source_lists_the_ones_that_exist():
    with pytest.raises(KeyError) as exc:
        probe(w=FakeSql()).tables("typo")
    assert "w" in str(exc.value)


def test_closing_swallows_a_raising_retriever():
    """A probe that cannot shut down cleanly must not lose the measurement it just made."""

    class Bad(FakeSql):
        async def close(self):
            raise RuntimeError("no")

    run(probe(b=Bad()).close())


# ------------------------------------------------------------------------------- cli


def test_the_cli_exposes_every_measurement():
    """A helper nobody can invoke from a shell is a helper that gets re-hand-rolled."""
    parser = pack_probe.build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    commands = set(actions[0].choices)
    assert {
        "sources",
        "leaves",
        "count",
        "population",
        "values",
        "spread",
        "selectivity",
        "pair",
        "cost",
        "sql",
        "dsl",
    } <= commands


def test_the_module_names_no_backend_coordinate():
    """The reason this is a library and not another script: no host, no CA, no env var.

    Every hand-rolled probe pinned its own URL, its own certificate bundle and its own
    credential env names, so it measured a path the pipeline does not take. If one appears
    here, the library has become the thing it replaced.
    """
    import pathlib

    text = pathlib.Path(pack_probe.__file__).read_text(encoding="utf-8")
    assert "https://" not in text
    assert ".pem" not in text
    assert "getenv" not in text and "environ" not in text
