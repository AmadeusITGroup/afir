"""
Tests for per-source timeout + progress reporting in LogRetrievalEngine._gather.

We build an engine with no retrievers wired (the pack/backends are empty) and inject
fake retrievers directly, so no LLM / ES / Databricks access is needed.
"""

import asyncio

import pytest

from src.log_retrieval import LogRetrievalEngine
from src.models.pydantic_models import RetrievalQuery


class _FakeRetriever:
    def __init__(self, delay, rows):
        self.delay = delay
        self.rows = rows
        self.cancelled = False

    async def retrieve(self, query):
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.rows


def _engine(retrievers, timeout=0.2):
    # knowledge_pack=None + empty config -> no retrievers built in __init__.
    engine = LogRetrievalEngine(
        {"per_source_timeout_seconds": timeout}, llm_client=None, knowledge_pack=None
    )
    engine.retrievers = retrievers
    return engine


def _q(source):
    return RetrievalQuery(
        target_log_source=source,
        natural_language_query="q",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )


async def test_slow_source_times_out_fast_sibling_returned():
    slow = _FakeRetriever(delay=100, rows=[{"slow": 1}])
    fast = _FakeRetriever(delay=0, rows=[{"fast": 1}])
    engine = _engine({"slow": slow, "fast": fast}, timeout=0.15)

    events = []
    logs = await engine.retrieve(
        [_q("slow"), _q("fast")],
        progress_cb=lambda s, st, m: events.append((s, st)),
    )

    # Fast source returned; slow source omitted (timed out) — not a whole-run hang.
    assert logs == {"fast": [{"fast": 1}]}
    assert ("fast", "completed") in events
    assert ("slow", "timeout") in events
    # The timed-out task was cancelled + drained.
    assert slow.cancelled is True


async def test_failed_source_reported_and_omitted():
    class _Boom:
        async def retrieve(self, query):
            raise RuntimeError("backend down")

    engine = _engine({"boom": _Boom()}, timeout=1)
    events = []
    logs = await engine.retrieve(
        [_q("boom")], progress_cb=lambda s, st, m: events.append((s, st))
    )
    assert logs == {}
    assert ("boom", "failed") in events


async def test_outer_cancel_cancels_subtasks():
    slow = _FakeRetriever(delay=100, rows=[])
    engine = _engine({"slow": slow}, timeout=100)

    task = asyncio.ensure_future(engine.retrieve([_q("slow")]))
    await asyncio.sleep(0.05)  # let the retrieval start
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slow.cancelled is True


# ── opt-in extended retrieval timeout ──────────────────────────────────────


def test_source_timeout_default_vs_extended():
    """Extended mode raises the cap (never lowers it); default leaves it untouched."""
    engine = _engine({}, timeout=20)
    # No per-source config: normal cap is the global default, extended is 4x.
    assert engine._source_timeout("x", 20, extended=False) == 20
    assert engine._source_timeout("x", 20, extended=True) == 80

    # Global extended value wins when set.
    engine.config["extended_retrieval_timeout_seconds"] = 900
    assert engine._source_timeout("x", 20, extended=True) == 900
    # ...but default mode still ignores it.
    assert engine._source_timeout("x", 20, extended=False) == 20


def test_source_timeout_per_source_extended_override():
    """A source's own extended_retrieval_timeout_seconds beats the global one."""

    class _R:
        config = {
            "retrieval_timeout_seconds": 330,
            "extended_retrieval_timeout_seconds": 1200,
        }

    engine = _engine({"db": _R()}, timeout=20)
    engine.config["extended_retrieval_timeout_seconds"] = 900
    # Normal mode: the source's normal override (330).
    assert engine._source_timeout("db", 20, extended=False) == 330
    # Extended mode: the source's extended override (1200) beats the global 900.
    assert engine._source_timeout("db", 20, extended=True) == 1200


async def test_extended_lets_slow_source_complete():
    """A source slower than the normal cap times out by default but completes when
    the caller opts into extended mode."""
    # delay 0.3s; normal cap 0.15s -> would time out; extended 4x = 0.6s -> completes.
    slow = _FakeRetriever(delay=0.3, rows=[{"ok": 1}])

    engine = _engine({"slow": slow}, timeout=0.15)
    default_logs = await engine.retrieve([_q("slow")])
    assert default_logs == {}  # timed out under the normal cap

    slow2 = _FakeRetriever(delay=0.3, rows=[{"ok": 1}])
    engine2 = _engine({"slow": slow2}, timeout=0.15)
    ext_logs = await engine2.retrieve([_q("slow")], extended=True)
    assert ext_logs == {"slow": [{"ok": 1}]}  # completed under the extended cap


async def test_extended_widens_databricks_poll_budget_and_restores():
    """Extended mode bumps a Databricks-like retriever's max_poll_attempts to cover the
    extended cap, then restores the original value after the run."""

    class _DBLike:
        # Duck-typed like DatabricksRetriever: has a poll budget the engine widens.
        max_poll_attempts = 60
        poll_interval = 5

        def __init__(self):
            self.seen_budget = None
            self.config = {"extended_retrieval_timeout_seconds": 900}

        async def retrieve(self, query):
            self.seen_budget = self.max_poll_attempts
            return [{"ok": 1}]

    db = _DBLike()
    engine = _engine({"db": db}, timeout=20)
    await engine.retrieve([_q("db")], extended=True)

    # During the run the budget covered the 900s cap (900/5 + 2 = 182 attempts)...
    assert db.seen_budget == 182
    # ...and it was restored afterwards.
    assert db.max_poll_attempts == 60


async def test_default_mode_never_touches_poll_budget():
    class _DBLike:
        max_poll_attempts = 60
        poll_interval = 5
        config = {}

        async def retrieve(self, query):
            return [{"ok": 1}]

    db = _DBLike()
    engine = _engine({"db": db}, timeout=1)
    await engine.retrieve([_q("db")])  # extended defaults to False
    assert db.max_poll_attempts == 60


# ── pack-declared primary sources get a much larger budget ──────────────
# One number lands on both coupled budgets: the engine's per-source cap and
# the retriever's own statement budget. Whichever is lower wins silently.


def _primary_budget(merged, config=None):
    engine = _engine({}, timeout=20)
    engine.config.update(config or {})
    engine._apply_primary_budget(merged)
    return merged


def test_primary_class_raises_both_coupled_budgets_from_one_number():
    merged = _primary_budget(
        {
            "name": "record_lake",
            "retrieval_class": "primary",
            # What the backend block says — shared with the cheap lookups beside it.
            "retrieval_timeout_seconds": 1800,
            "statement_timeout_seconds": 1800,
            "poll_interval_seconds": 5,
            "max_poll_attempts": 360,
        },
        {"primary_source_timeout_seconds": 3000},
    )
    assert merged["retrieval_timeout_seconds"] == 3000
    # Both budgets must move: raising only the engine cap leaves the statement cancelled at the old number.
    assert merged["statement_timeout_seconds"] == 3000
    # ...and the legacy attempt-count floor cannot cut the wall clock short (3000/5 + 2).
    assert merged["max_poll_attempts"] == 602


def test_a_source_with_no_retrieval_class_keeps_its_backend_budget():
    merged = _primary_budget(
        {
            "name": "automation_registry",
            "retrieval_timeout_seconds": 1800,
            "statement_timeout_seconds": 1800,
            "max_poll_attempts": 360,
        },
        {"primary_source_timeout_seconds": 3000},
    )
    assert merged["retrieval_timeout_seconds"] == 1800
    assert merged["statement_timeout_seconds"] == 1800
    assert merged["max_poll_attempts"] == 360


def test_primary_budget_never_lowers_an_explicit_value():
    """Raising only; a backend configured above the primary budget keeps its value."""
    merged = _primary_budget(
        {
            "name": "record_document_scope_sweep",
            "retrieval_class": "primary",
            "retrieval_timeout_seconds": 5400,
            "statement_timeout_seconds": 5400,
            "max_poll_attempts": 2000,
        },
        {"primary_source_timeout_seconds": 3000},
    )
    assert merged["retrieval_timeout_seconds"] == 5400
    assert merged["statement_timeout_seconds"] == 5400
    assert merged["max_poll_attempts"] == 2000


def test_primary_budget_falls_back_to_the_engine_default_with_no_config_key():
    """The config predates this key; the default must be non-zero and above the ordinary per-source cap."""
    merged = _primary_budget({"name": "p", "retrieval_class": "primary"})
    assert merged["retrieval_timeout_seconds"] == LogRetrievalEngine._PRIMARY_TIMEOUT_SECONDS
    assert merged["statement_timeout_seconds"] == LogRetrievalEngine._PRIMARY_TIMEOUT_SECONDS


def test_per_source_primary_override_beats_the_global_one():
    merged = _primary_budget(
        {
            "name": "p",
            "retrieval_class": "PRIMARY",  # case-insensitive: a pack is hand-written
            "primary_retrieval_timeout_seconds": 4200,
        },
        {"primary_source_timeout_seconds": 3000},
    )
    assert merged["retrieval_timeout_seconds"] == 4200


def test_the_engine_cap_reads_the_budget_primary_wrote():
    """`_apply_primary_budget` writes `retrieval_timeout_seconds`; `_source_timeout` reads it.
    Computing the number in both places is how the two budgets drift apart."""

    class _R:
        config = {}

    r = _R()
    engine = _engine({"p": r}, timeout=20)
    assert engine._source_timeout("p", 20) == 20  # before: the ordinary default
    r.config = _primary_budget(
        {"name": "p", "retrieval_class": "primary"},
        {"primary_source_timeout_seconds": 3000},
    )
    assert engine._source_timeout("p", 20) == 3000


# ── the primary budget is editable in the running process ───────────────
# `primary_source_timeout_seconds` is marked `live` in the UI; the engine holds the
# config by reference and `refresh_primary_budgets` re-applies it on each live edit.


def test_refresh_rewrites_a_primary_sources_budget_from_the_current_config():
    class _R:
        def __init__(self):
            self.config = {"name": "record_lake", "retrieval_class": "primary"}

    r = _R()
    engine = _engine({"record_lake": r}, timeout=20)
    engine.config["primary_source_timeout_seconds"] = 3000
    engine._apply_primary_budget(r.config)
    assert engine._source_timeout("record_lake", 20) == 3000

    # The operator edits the field; the live dict is the one the engine holds.
    engine.config["primary_source_timeout_seconds"] = 7200
    assert engine.refresh_primary_budgets() == 1
    assert r.config["retrieval_timeout_seconds"] == 7200
    # Both budgets move, as at build time; the lower wins silently.
    assert r.config["statement_timeout_seconds"] == 7200
    assert engine._source_timeout("record_lake", 20) == 7200


def test_refresh_can_LOWER_the_budget_it_previously_wrote():
    """Only lowers values this mechanism previously wrote; operator-configured values are unchanged.
    """
    class _R:
        config = {"name": "p", "retrieval_class": "primary"}

    r = _R()
    engine = _engine({"p": r}, timeout=20)
    engine.config["primary_source_timeout_seconds"] = 7200
    engine._apply_primary_budget(r.config)
    assert r.config["retrieval_timeout_seconds"] == 7200

    engine.config["primary_source_timeout_seconds"] = 1800
    engine.refresh_primary_budgets()
    assert r.config["retrieval_timeout_seconds"] == 1800
    assert r.config["statement_timeout_seconds"] == 1800
    # The attempt-count floor follows it down too, or it becomes the real cap (1800/5 + 2).
    assert r.config["max_poll_attempts"] == 362


def test_refresh_still_will_not_lower_a_value_the_operator_configured():
    """Operator-configured values above the primary budget survive a refresh unchanged."""

    class _R:
        config = {
            "name": "p",
            "retrieval_class": "primary",
            "retrieval_timeout_seconds": 9000,
            "statement_timeout_seconds": 9000,
        }

    r = _R()
    engine = _engine({"p": r}, timeout=20)
    engine.config["primary_source_timeout_seconds"] = 7200
    engine._apply_primary_budget(r.config)
    engine.config["primary_source_timeout_seconds"] = 1800
    engine.refresh_primary_budgets()
    assert r.config["retrieval_timeout_seconds"] == 9000
    assert r.config["statement_timeout_seconds"] == 9000


def test_refresh_leaves_non_primary_sources_alone():
    """Only primary-class retrievers are touched; a refresh must not extend other sources."""

    class _R:
        def __init__(self, cfg):
            self.config = cfg

    primary = _R({"name": "p", "retrieval_class": "primary"})
    ordinary = _R({"name": "o", "retrieval_timeout_seconds": 20})
    engine = _engine({"p": primary, "o": ordinary}, timeout=20)
    engine.config["primary_source_timeout_seconds"] = 7200
    assert engine.refresh_primary_budgets() == 1
    assert ordinary.config == {"name": "o", "retrieval_timeout_seconds": 20}


def test_the_shipped_default_is_two_hours():
    """Pins the shipped default; the config predating this key must fall back to a safe non-zero value."""
    assert LogRetrievalEngine._PRIMARY_TIMEOUT_SECONDS == 7200


# ── the query must be reportable before it is answered ─────────────────
# `last_generated_query` is assigned before the backend call so a timed-out source
# still publishes its query text.


class _AnnouncingRetriever:
    """A retriever that publishes its query, then takes a while to answer."""

    def __init__(self, text="SELECT 1", delay=0, rows=None, fail=False):
        self.text = text
        self.delay = delay
        self.rows = rows if rows is not None else [{"r": 1}]
        self.fail = fail
        self.last_generated_query = None

    async def retrieve(self, query, guidance="", on_query=None):
        from src.retrievers.base import publish_query

        publish_query(self, self.text, on_query)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("backend said no")
        return self.rows


def _query_events(events):
    return [(s, m) for s, st, m in events if st == "query_ready"]


async def test_the_query_is_announced_before_the_rows_come_back():
    r = _AnnouncingRetriever(delay=0.05)
    engine = _engine({"s": r}, timeout=5)
    events = []
    await engine.retrieve(
        [_q("s")], progress_cb=lambda s, st, m: events.append((s, st, m))
    )
    statuses = [st for _s, st, _m in events]
    # query_ready must precede the terminal status.
    assert statuses.index("query_ready") < statuses.index("completed")
    assert r.last_generated_query == "SELECT 1"


async def test_a_source_that_times_out_still_reported_its_query():
    """A timeout must not suppress the query text."""
    r = _AnnouncingRetriever(text="SELECT slow", delay=100)
    engine = _engine({"s": r}, timeout=0.15)
    events = []
    logs = await engine.retrieve(
        [_q("s")], progress_cb=lambda s, st, m: events.append((s, st, m))
    )
    assert "s" not in logs  # it did time out
    assert _query_events(events)  # and its query was published anyway
    assert r.last_generated_query == "SELECT slow"


async def test_a_source_that_fails_still_reported_its_query():
    r = _AnnouncingRetriever(text="SELECT boom", fail=True)
    engine = _engine({"s": r}, timeout=5)
    events = []
    await engine.retrieve(
        [_q("s")], progress_cb=lambda s, st, m: events.append((s, st, m))
    )
    assert [st for _s, st, _m in events if st == "failed"]
    assert r.last_generated_query == "SELECT boom"


async def test_a_retriever_that_does_not_accept_on_query_still_retrieves():
    """`on_query` is observability; failing to deliver it must not fail the retrieval."""
    legacy = _FakeRetriever(delay=0, rows=[{"ok": 1}])  # retrieve(self, query) only
    engine = _engine({"s": legacy}, timeout=5)
    logs = await engine.retrieve([_q("s")])
    assert logs["s"] == [{"ok": 1}]


async def test_a_raising_announcement_callback_does_not_break_retrieval():
    r = _AnnouncingRetriever()
    engine = _engine({"s": r}, timeout=5)

    def boom(source, status, message):
        raise RuntimeError("the UI listener misbehaved")

    logs = await engine.retrieve([_q("s")], progress_cb=boom)
    assert logs["s"] == [{"r": 1}]


async def test_each_source_announces_its_own_query_not_a_siblings():
    """Each source has its own callback; a late announcement must name the source it came from."""
    a = _AnnouncingRetriever(text="SELECT a", delay=0.05)
    b = _AnnouncingRetriever(text="SELECT b", delay=0)
    engine = _engine({"a": a, "b": b}, timeout=5)
    events = []
    await engine.retrieve(
        [_q("a"), _q("b")], progress_cb=lambda s, st, m: events.append((s, st, m))
    )
    announced = {s for s, _m in _query_events(events)}
    assert announced == {"a", "b"}


# --- a source that was asked and did not answer ----------------------------------------
# A source that did not answer is absent from `logs`; a source that answered with nothing
# is present and empty. The two mean different things to every consumer.


async def test_a_timed_out_source_is_named_with_its_budget():
    slow = _FakeRetriever(delay=100, rows=[{"slow": 1}])
    fast = _FakeRetriever(delay=0, rows=[{"fast": 1}])
    engine = _engine({"slow": slow, "fast": fast}, timeout=0.15)

    unanswered = {}
    logs = await engine.retrieve(
        [_q("slow"), _q("fast")], unanswered_out=unanswered
    )

    assert logs == {"fast": [{"fast": 1}]}
    # Only unanswered sources appear in the dict; an answering source must not be listed.
    assert list(unanswered) == ["slow"]
    assert "0.15s" in unanswered["slow"]


async def test_a_failed_source_carries_its_exception_TYPE_and_not_its_message():
    class _Boom:
        async def retrieve(self, query):
            raise RuntimeError("host=db.internal user=svc_acct password=hunter2")

    engine = _engine({"boom": _Boom()}, timeout=1)
    unanswered = {}
    assert await engine.retrieve([_q("boom")], unanswered_out=unanswered) == {}
    # Names the exception type, not the message; backend text may carry credentials.
    assert "RuntimeError" in unanswered["boom"]
    assert "hunter2" not in unanswered["boom"]


async def test_an_empty_answer_is_not_a_non_answer():
    """The whole point of the dict: zero rows and no rows are different facts."""
    engine = _engine({"empty": _FakeRetriever(delay=0, rows=[])}, timeout=1)
    unanswered = {}
    logs = await engine.retrieve([_q("empty")], unanswered_out=unanswered)
    assert logs == {"empty": []}
    assert unanswered == {}


async def test_a_later_answer_RETRACTS_an_earlier_non_answer():
    """A successful follow-up pass must remove the source from the non-answer dict."""
    engine = _engine({"src": _FakeRetriever(delay=100, rows=[{"r": 1}])}, timeout=0.15)
    unanswered = {}
    await engine.retrieve([_q("src")], unanswered_out=unanswered)
    assert "src" in unanswered

    engine.retrievers = {"src": _FakeRetriever(delay=0, rows=[{"r": 1}])}
    logs = await engine.retrieve([_q("src")], unanswered_out=unanswered)
    assert logs == {"src": [{"r": 1}]}
    assert unanswered == {}


async def test_a_cancelled_source_says_so_rather_than_nothing():
    async def _run(unanswered):
        engine = _engine({"slow": _FakeRetriever(delay=100, rows=[])}, timeout=50)
        await engine.retrieve([_q("slow")], unanswered_out=unanswered)

    unanswered = {}
    task = asyncio.ensure_future(_run(unanswered))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "cancel" in unanswered["slow"].lower()


async def test_omitting_the_out_param_leaves_retrieval_byte_identical():
    """Every existing caller passes nothing; the argument must be inert when absent."""
    engine = _engine(
        {"slow": _FakeRetriever(delay=100, rows=[]), "ok": _FakeRetriever(0, [{"a": 1}])},
        timeout=0.15,
    )
    assert await engine.retrieve([_q("slow"), _q("ok")]) == {"ok": [{"a": 1}]}


# --- a source asked with a placeholder -----------------------------------------------
# A placeholder predicate matches nothing; the backend returns zero rows and every consumer
# reads that as the source's answer — a `zero_rows` declaration can even make it decisive. So
# an empty result from such a query is converted to a non-answer at retrieval time.


async def test_an_empty_result_from_a_placeholder_query_is_a_NON_answer():
    r = _AnnouncingRetriever(
        text=(
            "SELECT * FROM record WHERE creator.office_id = '<unitId>' "
            "AND creator.sign.red LIKE '<sign>%'"
        ),
        rows=[],
    )
    engine = _engine({"sweep": r}, timeout=5)
    events = []
    unanswered = {}
    logs = await engine.retrieve(
        [_q("sweep")],
        progress_cb=lambda s, st, m: events.append((s, st, m)),
        unanswered_out=unanswered,
    )

    # ABSENT from logs, not present-and-empty: the two shapes mean different things, and
    # only the absent one means "we learned nothing about this source".
    assert logs == {}
    assert "'<unitId>'" in unanswered["sweep"]
    assert "'<sign>%'" in unanswered["sweep"]
    # And the operator's own line says which one it was, in the direction that matters:
    # the QUERY matched nothing, the source did not answer with nothing.
    failed = [m for _s, st, m in events if st == "failed"]
    assert failed and "placeholder" in failed[0]
    assert "completed" not in [st for _s, st, _m in events]


async def test_a_placeholder_query_that_RETURNED_ROWS_is_left_alone():
    """The bound, and it is the whole reason this is not a pre-execution refusal: a
    placeholder can sit somewhere harmless — a projected label, an `ORDER BY` — and rows
    that came back are real rows whatever the text looked like. Converting those would
    delete evidence; converting an empty result deletes nothing."""
    r = _AnnouncingRetriever(
        text="SELECT '<label>' AS tag, * FROM t WHERE office = 'NNN1P15CD'",
        rows=[{"tag": "<label>", "office": "NNN1P15CD"}],
    )
    engine = _engine({"src": r}, timeout=5)
    events = []
    unanswered = {}
    logs = await engine.retrieve(
        [_q("src")],
        progress_cb=lambda s, st, m: events.append((s, st, m)),
        unanswered_out=unanswered,
    )

    assert logs == {"src": [{"tag": "<label>", "office": "NNN1P15CD"}]}
    assert unanswered == {}
    assert "completed" in [st for _s, st, _m in events]


async def test_an_empty_result_from_a_CLEAN_query_stays_an_answer():
    """The other half of the bound: the conversion is keyed on the placeholder, not on
    the emptiness. A source that was asked properly and had nothing to say still ANSWERS,
    because that is a finding a pack can declare decisive."""
    r = _AnnouncingRetriever(text="SELECT * FROM t WHERE office = 'NNN1P15CD'", rows=[])
    engine = _engine({"src": r}, timeout=5)
    unanswered = {}
    logs = await engine.retrieve([_q("src")], unanswered_out=unanswered)
    assert logs == {"src": []}
    assert unanswered == {}


async def test_a_placeholder_source_is_absent_even_with_no_out_param():
    """The omission from `logs` is not a courtesy of the out-parameter — a caller that
    passes nothing must still not be handed an empty list that means nothing."""
    engine = _engine(
        {
            "sweep": _AnnouncingRetriever(text="WHERE a = '<x>'", rows=[]),
            "ok": _AnnouncingRetriever(text="WHERE a = 'real'", rows=[{"a": 1}]),
        },
        timeout=5,
    )
    assert await engine.retrieve([_q("sweep"), _q("ok")]) == {"ok": [{"a": 1}]}
