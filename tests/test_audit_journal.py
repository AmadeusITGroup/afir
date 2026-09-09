"""
The access journal: what it records, and the four ways it may not fail while doing it.

Every assertion here is about a property that fails SILENTLY in production. A journal that
records nothing looks exactly like a deployment nobody uses; one that buffers forever looks
like a working one until the process dies with the record inside it; one that raises inside
the middleware turns a page load into a 500. So the tests are grouped by the failure they
would otherwise let through, and the writes go through ``LocalStorage`` rather than a mock —
it is the seam's oracle, and the property that matters (a day's entries survive a restart) is
a claim about bytes on disk, not about calls.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.audit_journal import (JOURNAL_PREFIX, AuditJournal,
                               build_audit_journal, day_key)
from src.identity import LOCAL_IDENTITY, Identity
from src.storage import LocalStorage


@pytest.fixture()
def store(tmp_path):
    return LocalStorage(tmp_path / "state")


@pytest.fixture()
def journal(store):
    return AuditJournal(storage=store, enabled=True, flush_seconds=3600)


def _entries(store, key=None):
    blob = store.get_text(key or day_key()) or ""
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


def _identity(name="someone@example.com", role="user", source="token"):
    return Identity(
        user_id="42", user_name=name, role=role, source=source, groups=["g"],
        role_reason="test",
    )


# -- what lands ------------------------------------------------------------


def test_a_request_is_recorded_with_who_asked(journal, store):
    journal.record_request(_identity(), "GET", "/api/v1/jobs", 200, 12.5)
    assert journal.flush_now() == 1
    (entry,) = _entries(store)
    assert entry["kind"] == "request"
    assert entry["user"] == "someone@example.com"
    assert entry["role"] == "user"
    assert (entry["method"], entry["path"], entry["status"]) == ("GET", "/api/v1/jobs", 200)
    assert entry["ms"] == 12.5
    assert entry["at"]


def test_how_the_identity_was_established_is_recorded(journal, store):
    """A `local` entry must never read as an authenticated one — the whole journal's worth
    depends on being able to tell an asserted caller from the single-operator fallback."""
    journal.record_request(LOCAL_IDENTITY, "GET", "/api/v1/jobs", 200, 1.0)
    journal.record_request(_identity(source="header"), "GET", "/api/v1/jobs", 200, 1.0)
    journal.flush_now()
    assert [e["auth"] for e in _entries(store)] == ["local", "header"]


def test_an_entry_with_no_identity_says_so_rather_than_leaving_it_blank(journal, store):
    """`auth` is on every entry, so its ABSENCE would mean "the writer forgot" — a different
    claim from "nobody was identified", which is what a refusal and an unauthenticated path
    both are."""
    journal.record("visit", None, path="/")
    journal.flush_now()
    (entry,) = _entries(store)
    assert entry["auth"] == "unresolved"
    assert "user" not in entry


def test_a_refusal_is_recorded_although_it_has_no_identity(journal, store):
    """The one record of why a caller sees a bare 403; by construction there is no caller."""
    journal.record("refused", None, method="GET", path="/api/v1/jobs", status=403,
                   detail="two conflicting user headers")
    journal.flush_now()
    (entry,) = _entries(store)
    assert entry["kind"] == "refused" and "conflicting" in entry["detail"]
    assert "user" not in entry


def test_a_config_change_records_the_paths(journal, store):
    journal.record("config_change", _identity(role="admin"), target="base",
                   changed=["jobs.history_max_items"], values={"jobs.history_max_items": 500})
    journal.flush_now()
    (entry,) = _entries(store)
    assert entry["changed"] == ["jobs.history_max_items"]
    assert entry["values"] == {"jobs.history_max_items": 500}
    assert entry["target"] == "base"


def test_the_health_probe_is_excluded(journal, store):
    """It fires every few seconds forever and would drown everything that carries meaning."""
    journal.record_request(_identity(), "GET", "/health", 200, 1.0)
    journal.record_request(_identity(), "GET", "/api/v1/jobs", 200, 1.0)
    journal.flush_now()
    assert [e["path"] for e in _entries(store)] == ["/api/v1/jobs"]


def test_entries_land_in_one_object_per_day(journal, store):
    journal.record("request", _identity(), path="/a")
    journal.flush_now()
    journal.record("request", _identity(), path="/b")
    journal.flush_now()
    keys = [obj.key for obj in store.list_keys(JOURNAL_PREFIX)]
    assert keys == [day_key()]
    assert [e["path"] for e in _entries(store)] == ["/a", "/b"]


# -- may not fail ----------------------------------------------------------


def test_disabled_without_a_store_and_records_nothing():
    """No store is not a soft failure: buffering into a process about to restart is worse."""
    j = AuditJournal(storage=None, enabled=True)
    j.record_request(_identity(), "GET", "/x", 200, 1.0)
    assert j.enabled is False
    assert j.pending == 0
    assert j.flush_now() == 0
    assert j.tail() == []


def test_a_refusing_sink_never_raises_and_is_reported_once(store, caplog):
    def refuse(key, text):
        raise OSError("volume is read-only")

    store.append_text = refuse
    j = AuditJournal(storage=store, enabled=True, flush_seconds=3600)
    with caplog.at_level("ERROR"):
        for _ in range(3):
            j.record_request(_identity(), "GET", "/x", 200, 1.0)
            assert j.flush_now() == 0
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1, "a failing sink must not log per entry"
    assert "runs are unaffected" in errors[0].getMessage()
    assert j.stats()["write_failures"] == 3


def test_a_full_buffer_drops_the_oldest_and_says_so(store):
    """The alternative is growing until the process dies, taking the investigation with it."""
    j = AuditJournal(storage=store, enabled=True, flush_seconds=3600, max_buffer=2,
                     max_pending=2)
    for i in range(5):
        j.record("request", _identity(), path=f"/{i}")
    assert j.pending == 2
    assert j.dropped == 3
    j.flush_now()
    entries = _entries(store)
    assert [e.get("path") for e in entries[:2]] == ["/3", "/4"]
    assert entries[-1]["kind"] == "journal_overflow" and entries[-1]["dropped"] == 3
    # Reset once reported, or every later batch re-reports the same loss.
    assert j.dropped == 0


def test_an_unserialisable_field_does_not_lose_the_batch(journal, store):
    journal.record("request", _identity(), path="/x", extra=object())
    assert journal.flush_now() == 1
    assert _entries(store)[0]["path"] == "/x"


def test_a_recorded_entry_is_not_lost_by_a_concurrent_flush(store):
    """`flush_now` snapshots rather than clearing afterwards, so a write in flight cannot
    swallow an entry recorded while it ran."""
    j = AuditJournal(storage=store, enabled=True, flush_seconds=3600)
    calls = []
    real = store.append_text

    def appending(key, text):
        calls.append(text)
        if len(calls) == 1:
            j.record("request", _identity(), path="/during")
        return real(key, text)

    store.append_text = appending
    j.record("request", _identity(), path="/first")
    j.flush_now()
    assert j.pending == 1
    j.flush_now()
    assert [e["path"] for e in _entries(store)] == ["/first", "/during"]


def test_flush_is_due_on_the_buffer_size_before_the_timer(store):
    j = AuditJournal(storage=store, enabled=True, flush_seconds=3600, max_buffer=2)
    j.record("request", _identity(), path="/a")
    assert j.due() is False
    j.record("request", _identity(), path="/b")
    assert j.due() is True


async def test_stop_drains_what_is_buffered(journal, store):
    journal.record("request", _identity(), path="/last")
    assert await journal.stop() == 1
    assert [e["path"] for e in _entries(store)] == ["/last"]


# -- reading back ----------------------------------------------------------


def test_tail_is_newest_first_and_bounded(journal, store):
    for i in range(5):
        journal.record("request", _identity(), path=f"/{i}")
    journal.flush_now()
    assert [e["path"] for e in journal.tail(limit=2)] == ["/4", "/3"]


def test_tail_filters_by_user_case_insensitively(journal, store):
    journal.record("request", _identity("Alice@x"), path="/a")
    journal.record("request", _identity("bob@x"), path="/b")
    journal.flush_now()
    assert [e["path"] for e in journal.tail(user="alice@X")] == ["/a"]


def test_tail_reads_days_newest_first_and_stops_at_since(journal, store):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    store.append_text(
        day_key(yesterday),
        json.dumps({"at": yesterday.isoformat(), "kind": "request", "path": "/old"}) + "\n",
    )
    journal.record("request", _identity(), path="/new")
    journal.flush_now()
    both = [e["path"] for e in journal.tail()]
    assert both == ["/new", "/old"]
    cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert [e["path"] for e in journal.tail(since=cutoff)] == ["/new"]


def test_a_corrupt_line_is_skipped_rather_than_fatal(journal, store):
    store.append_text(day_key(), "not json\n" + json.dumps({"at": "z", "kind": "request"}) + "\n")
    assert len(journal.tail()) == 1


def test_tail_survives_an_unreadable_day(journal, store):
    journal.record("request", _identity(), path="/x")
    journal.flush_now()

    def refuse(key):
        raise OSError("gone")

    store.get_text = refuse
    assert journal.tail() == []


# -- housekeeping ----------------------------------------------------------


def test_prune_compares_the_date_in_the_key(journal, store):
    """Not the mtime: an appended file's mtime is the last WRITE, so today's object looks
    freshly created every day and an old one looks new the moment it is read back."""
    old = datetime.now(timezone.utc) - timedelta(days=120)
    store.append_text(day_key(old), '{"kind":"request"}\n')
    journal.record("request", _identity(), path="/x")
    journal.flush_now()
    assert journal.prune() == 1
    assert [obj.key for obj in store.list_keys(JOURNAL_PREFIX)] == [day_key()]


def test_prune_keeps_everything_when_retention_is_zero(store):
    j = AuditJournal(storage=store, enabled=True, retention_days=0)
    store.append_text(day_key(datetime.now(timezone.utc) - timedelta(days=999)), "{}\n")
    assert j.prune() == 0


def test_stats_report_the_journals_own_state(journal, store):
    journal.record("request", _identity(), path="/x")
    stats = journal.stats()
    assert stats["enabled"] is True and stats["pending"] == 1
    journal.flush_now()
    assert journal.stats()["days_held"] == 1


# -- configuration ---------------------------------------------------------


def test_auto_follows_the_platform(store, monkeypatch):
    """On where an ingress names callers; off on a laptop, where the journal would record
    one person visiting their own machine."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    assert build_audit_journal({}, store).enabled is False
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    assert build_audit_journal({}, store).enabled is True


def test_explicit_on_beats_the_platform(store, monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    assert build_audit_journal({"audit": {"enabled": "on"}}, store).enabled is True
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    assert build_audit_journal({"audit": {"enabled": "off"}}, store).enabled is False


def test_excluded_paths_are_read_from_a_scalar(store):
    """The Configuration tab writes a comma-separated string, never a YAML list."""
    j = build_audit_journal(
        {"audit": {"enabled": "on", "exclude_paths": "/health, /docs"}}, store
    )
    assert j.excluded_paths == frozenset({"/health", "/docs"})


def test_a_typo_falls_back_to_the_default_rather_than_zero(store):
    """A `flush_seconds: ""` from a half-finished edit must not mean "flush never" or
    "flush every zero seconds"."""
    j = build_audit_journal(
        {"audit": {"enabled": "on", "flush_seconds": "", "max_buffer": "abc"}}, store
    )
    assert j.flush_seconds == 30.0 and j.max_buffer == 200


def test_the_pending_ceiling_is_never_below_the_batch_size(store):
    j = AuditJournal(storage=store, enabled=True, max_buffer=500, max_pending=10)
    assert j.max_pending == 500
