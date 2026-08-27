"""
Tests for the retrieval cache: the store itself, and its wiring into
``LogRetrievalEngine._gather``.

Two halves, and the second is the one that matters. A cache is easy to get right in
isolation and easy to get wrong at the seam, because everything it must never keep is
already *absent* from the dict it stores from — so the integration tests assert the
absences (a timeout, a failure, a placeholder-empty) as well as the hit.

Engines are built with no retrievers wired (empty config, no pack) and fake retrievers
injected, so nothing here touches an LLM or a backend.
"""

import asyncio
import time

from src.log_retrieval import LogRetrievalEngine
from src.models.pydantic_models import ExtractedEntity, RetrievalQuery
from src.retrieval_cache import (
    DEFAULT_MAX_ENTRIES,
    DEFAULT_TTL_SECONDS,
    RetrievalCache,
    build_retrieval_cache,
    cache_key,
    cache_stats,
)


def _cache(**cfg):
    cfg.setdefault("enabled", True)
    return RetrievalCache(cfg)


def _q(source="src_a", question="q", **kw):
    kw.setdefault("date_from", "2024-01-01")
    kw.setdefault("date_to", "2024-01-02")
    return RetrievalQuery(
        target_log_source=source, natural_language_query=question, **kw
    )


# -- the key ------------------------------------------------------------------


def test_the_same_question_keys_the_same():
    assert cache_key(_q()) == cache_key(_q())


def test_entity_order_does_not_change_the_key():
    a = _q(
        entities=[
            ExtractedEntity(type="user", value="u1"),
            ExtractedEntity(type="office", value="o1"),
        ]
    )
    b = _q(
        entities=[
            ExtractedEntity(type="office", value="o1"),
            ExtractedEntity(type="user", value="u1"),
        ]
    )
    assert cache_key(a) == cache_key(b)


def test_a_different_entity_value_keys_differently():
    a = _q(entities=[ExtractedEntity(type="user", value="u1")])
    b = _q(entities=[ExtractedEntity(type="user", value="u2")])
    assert cache_key(a) != cache_key(b)


def test_a_raised_row_cap_misses_the_capped_entry():
    # A result at the cap is a floor, not an answer: re-serving a 500-row entry to a run
    # configured for 5000 keeps a truncation the operator has already paid to remove.
    assert cache_key(_q(), row_cap=500) != cache_key(_q(), row_cap=5000)


def test_analyst_guidance_keys_differently():
    # A rejected gate rewrites the query; the corrected ask must not hit the answer the
    # rejection was about.
    assert cache_key(_q(), guidance="") != cache_key(_q(), guidance="also check X")


def test_the_window_and_the_source_are_in_the_key():
    assert cache_key(_q(source="src_a")) != cache_key(_q(source="src_b"))
    assert cache_key(_q()) != cache_key(_q(date_to="2024-06-01"))


# -- the store ----------------------------------------------------------------


def test_a_disabled_cache_never_hits_and_never_keeps():
    cache = RetrievalCache({})
    assert cache.enabled is False
    assert cache.store("k", [{"a": 1}]) is False
    assert cache.get("k") is None


def test_an_answer_round_trips_with_everything_its_reader_must_state():
    cache = _cache()
    assert cache.store("k", [{"a": 1}], query="SELECT 1", key_enforced=True) is True
    hit = cache.get("k")
    assert hit is not None
    assert hit.rows == [{"a": 1}]
    assert hit.query == "SELECT 1"
    assert hit.key_enforced is True
    assert hit.age_seconds >= 0


def test_an_empty_answer_is_not_kept_by_default():
    cache = _cache()
    assert cache.store("k", []) is False
    assert cache.get("k") is None


def test_an_empty_answer_is_kept_where_a_deployment_says_so():
    cache = _cache(empty_ttl_seconds=60)
    assert cache.store("k", []) is True
    hit = cache.get("k")
    assert hit is not None
    assert hit.rows == []


def test_an_expired_entry_misses_and_is_dropped():
    cache = _cache(ttl_seconds=1)
    cache.store("k", [{"a": 1}])
    cache._entries["k"]["stored_at"] = time.time() - 5
    assert cache.get("k") is None
    assert cache.stats()["entries"] == 0
    assert cache.stats()["expiries"] == 1


def test_an_entry_that_cannot_be_dated_is_discarded():
    # Not read as brand new (which would serve it forever) and not as very old (which
    # would be indistinguishable from a miss): the timestamp is what makes an entry
    # expirable, so an entry nobody can date is dropped and said so.
    cache = _cache()
    cache.store("k", [{"a": 1}])
    cache._entries["k"]["stored_at"] = None
    assert cache.get("k") is None
    assert cache.stats()["entries"] == 0


def test_the_oldest_entry_is_evicted_at_the_entry_bound():
    cache = _cache(max_entries=2)
    cache.store("a", [{"n": 1}])
    cache.store("b", [{"n": 2}])
    cache.store("c", [{"n": 3}])
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert cache.stats()["evictions"] == 1


def test_a_read_makes_an_entry_the_most_recent():
    # The dict is only an LRU because `get` re-inserts; without that this evicts "a"
    # despite it being the entry in use.
    cache = _cache(max_entries=2)
    cache.store("a", [{"n": 1}])
    cache.store("b", [{"n": 2}])
    cache.get("a")
    cache.store("c", [{"n": 3}])
    assert cache.get("a") is not None
    assert cache.get("b") is None


def test_the_row_bound_evicts_and_the_totals_follow():
    cache = _cache(max_rows=5)
    cache.store("a", [{"n": i} for i in range(3)])
    cache.store("b", [{"n": i} for i in range(3)])
    assert cache.get("a") is None
    assert cache.stats()["rows"] == 3


def test_one_answer_over_the_whole_row_budget_is_refused_not_admitted():
    # Admitting it would evict every other entry to hold a single truncated source —
    # a cache that makes the next run slower.
    cache = _cache(max_rows=2)
    cache.store("a", [{"n": 1}])
    assert cache.store("big", [{"n": i} for i in range(10)]) is False
    assert cache.get("a") is not None
    assert cache.get("big") is None


def test_invalidate_drops_one_key_or_all_of_them():
    cache = _cache()
    cache.store("a", [{"n": 1}])
    cache.store("b", [{"n": 2}])
    assert cache.invalidate("a") == 1
    assert cache.invalidate("missing") == 0
    assert cache.invalidate() == 1
    assert cache.stats()["entries"] == 0
    assert cache.stats()["rows"] == 0


def test_stats_reports_the_denominator_beside_the_rate():
    cache = _cache()
    cache.store("a", [{"n": 1}])
    cache.get("a")
    cache.get("missing")
    stats = cache.stats()
    assert (stats["hits"], stats["misses"], stats["hit_rate"]) == (1, 1, 0.5)
    assert stats["stores"] == 1
    assert stats["hit_rate"] == 0.5


def test_a_bound_that_cannot_be_read_takes_its_default_not_zero():
    # Zero is the one value with its own meaning here (never cache), so a typo must not
    # become it.
    cache = _cache(ttl_seconds="soon", max_entries=None)
    assert cache.ttl_seconds == DEFAULT_TTL_SECONDS
    assert cache.max_entries == DEFAULT_MAX_ENTRIES


def test_describe_states_the_age_in_the_unit_a_reader_uses():
    cache = _cache()
    cache.store("k", [{"n": 1}])
    cache._entries["k"]["stored_at"] = time.time() - 240
    assert cache.get("k").describe() == "cached 4m ago"


# -- construction and reporting -----------------------------------------------


def test_a_deployment_declaring_nothing_gets_a_disabled_cache():
    assert build_retrieval_cache(None).enabled is False
    assert build_retrieval_cache({}).enabled is False
    assert build_retrieval_cache({"cache": {}}).enabled is False


def test_the_cache_is_built_from_the_log_sources_slice():
    cache = build_retrieval_cache({"cache": {"enabled": True, "ttl_seconds": 30}})
    assert cache.enabled is True
    assert cache.ttl_seconds == 30


def test_cache_stats_reports_not_wired_rather_than_raising():
    # The health endpoint that exists to report failures must not become one.
    class _Boom:
        class cache:  # noqa: N801 — a stand-in attribute, not a public class
            @staticmethod
            def stats():
                raise RuntimeError("nope")

    assert cache_stats(None) == (False, {})
    assert cache_stats(object()) == (False, {})
    assert cache_stats(_Boom()) == (False, {})
    wired, stats = cache_stats(LogRetrievalEngine({}, None, None))
    assert wired is True
    assert stats["enabled"] is False


# -- the seam: _gather --------------------------------------------------------


class _CountingRetriever:
    def __init__(self, rows, delay=0, boom=None, query_text="SELECT 1", keyed=False):
        self.rows = rows
        self.delay = delay
        self.boom = boom
        self.calls = 0
        self.last_generated_query = query_text
        self.last_key_enforced = keyed

    async def retrieve(self, query, on_query=None, **kw):
        self.calls += 1
        if on_query is not None:
            on_query(self.last_generated_query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.boom is not None:
            raise self.boom
        return self.rows


def _engine(retrievers, cache_cfg=None, timeout=0.2):
    config = {"per_source_timeout_seconds": timeout}
    if cache_cfg is not None:
        config["cache"] = cache_cfg
    engine = LogRetrievalEngine(config, llm_client=None, knowledge_pack=None)
    engine.retrievers = retrievers
    return engine


_ON = {"enabled": True, "ttl_seconds": 60}


async def test_a_second_identical_run_asks_the_source_nothing():
    r = _CountingRetriever([{"a": 1}])
    engine = _engine({"src_a": r}, cache_cfg=_ON)
    first = await engine.retrieve([_q()])
    second = await engine.retrieve([_q()])
    assert first == second == {"src_a": [{"a": 1}]}
    assert r.calls == 1


async def test_without_the_cache_the_second_run_re_queries():
    # The shipped default, and the baseline the cache must be byte-identical to.
    r = _CountingRetriever([{"a": 1}])
    engine = _engine({"src_a": r})
    await engine.retrieve([_q()])
    await engine.retrieve([_q()])
    assert r.calls == 2


async def test_a_served_answer_states_its_age_and_that_it_was_not_re_queried():
    # The row count a stage announces is read by the health scorer and narrated in the
    # report; a hit that looked like a fresh retrieval would date nothing.
    r = _CountingRetriever([{"a": 1}])
    engine = _engine({"src_a": r}, cache_cfg=_ON)
    await engine.retrieve([_q()])
    events = []
    await engine.retrieve([_q()], progress_cb=lambda s, st, m: events.append((st, m)))
    completed = [m for st, m in events if st == "completed"]
    assert len(completed) == 1
    assert "cached" in completed[0] and "not re-queried" in completed[0]
    assert "1 rows" in completed[0]


async def test_a_hit_replays_the_key_flag_rather_than_reading_a_shared_retriever():
    # Without this a cached keyed lookup reads as unkeyed and its zero-row FINDING
    # degrades to `unknown`.
    r = _CountingRetriever([{"a": 1}], keyed=True)
    engine = _engine({"src_a": r}, cache_cfg=_ON)
    await engine.retrieve([_q()], keyed_out={})
    r.last_key_enforced = False  # another job publishes an unkeyed query meanwhile
    keyed = {}
    queries = {}
    await engine.retrieve([_q()], keyed_out=keyed, queries_out=queries)
    assert keyed == {"src_a": True}
    assert queries == {"src_a": "SELECT 1"}


async def test_a_timed_out_source_is_never_cached():
    slow = _CountingRetriever([{"a": 1}], delay=100)
    engine = _engine({"src_a": slow}, cache_cfg=_ON, timeout=0.05)
    assert await engine.retrieve([_q()]) == {}
    await engine.retrieve([_q()])
    assert slow.calls == 2


async def test_a_failed_source_is_never_cached():
    boom = _CountingRetriever([], boom=RuntimeError("backend down"))
    engine = _engine({"src_a": boom}, cache_cfg=_ON)
    assert await engine.retrieve([_q()]) == {}
    await engine.retrieve([_q()])
    assert boom.calls == 2


async def test_an_empty_result_from_an_unfilled_placeholder_is_never_cached():
    # It is a non-answer, so it is absent from `logs` — which is what keeps it out of
    # the cache, with the store needing no idea the rule exists.
    r = _CountingRetriever([], query_text="WHERE office = '<office>'")
    engine = _engine({"src_a": r}, cache_cfg=dict(_ON, empty_ttl_seconds=60))
    assert await engine.retrieve([_q()]) == {}
    await engine.retrieve([_q()])
    assert r.calls == 2


async def test_extended_retrieval_bypasses_the_read_and_still_refreshes():
    # Extended mode is the operator asking again with a bigger budget; answering that
    # from a store answers a different question.
    r = _CountingRetriever([{"a": 1}])
    engine = _engine({"src_a": r}, cache_cfg=_ON)
    await engine.retrieve([_q()])
    await engine.retrieve([_q()], extended=True)
    assert r.calls == 2
    await engine.retrieve([_q()])
    assert r.calls == 2


async def test_a_corrected_ask_does_not_hit_the_uncorrected_answer():
    r = _CountingRetriever([{"a": 1}])
    engine = _engine({"src_a": r}, cache_cfg=_ON)
    await engine.retrieve([_q()])
    await engine.retrieve([_q()], guidance="also check the second office")
    assert r.calls == 2


async def test_one_source_hits_while_its_sibling_is_retrieved():
    a = _CountingRetriever([{"a": 1}])
    b = _CountingRetriever([{"b": 1}])
    engine = _engine({"src_a": a, "src_b": b}, cache_cfg=_ON)
    await engine.retrieve([_q(source="src_a")])
    logs = await engine.retrieve([_q(source="src_a"), _q(source="src_b")])
    assert logs == {"src_a": [{"a": 1}], "src_b": [{"b": 1}]}
    assert (a.calls, b.calls) == (1, 1)
