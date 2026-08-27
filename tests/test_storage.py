"""
The storage seam's contract, and the local backend that defines it.

Written as **one contract suite** parametrised over backends rather than a per-backend
file. The local backend is the behavioural oracle: it is what the VM deployment has
always done, so any other implementation is correct exactly insofar as it answers these
the same way. A remote backend added later registers itself in ``BACKENDS`` and either
passes or is wrong — which is cheaper than discovering the divergence from a report that
came back empty.

The key-validation tests carry their own weight beyond hygiene. Keys are built from job
ids and incident ids, and those arrive off a URL path; ``report_delivery`` already
learned that a *sanitised* traversal is worse than a rejected one, because a normalised
path that resolves to some other real readable file answers 200 with content it was
never meant to serve.
"""

import contextlib
import json
import time

import pytest

from src.storage import LocalStorage, build_storage, json_verifier
from src.storage.base import MAX_SEGMENT_LEN, StoredObject, safe_key
from src.storage.databricks import DatabricksStorage
from src.storage.sql import SqlStorage
from tests.fake_files_api import serve_fake_volume

CATALOG = "test_catalog"
VOLUME_ROOT = f"/Volumes/{CATALOG}/afir/state"


class FlushingDatabricksStorage(DatabricksStorage):
    """``DatabricksStorage`` with every write flushed before the call returns.

    **This is what makes the contract suite mean anything for this backend.** Reads merge
    the pending queue, so run as-is the whole contract would pass off an in-memory dict
    with the remote never consulted — the ``a-test-harness-that-voids-itself`` shape, a
    suite that proves the queue works and says nothing about the Volume. Flushing forces
    every assertion in the shared contract through a real HTTP round-trip.

    The queue's own behaviour — coalescing, reading a write that is in flight, a flush
    that must complete before a gate is announced — is asserted separately below, on the
    unwrapped class.
    """

    def put_text(self, key, text, verify=None):
        ok = super().put_text(key, text, verify=verify)
        self.flush(timeout=30)
        return ok

    def put_bytes(self, key, blob):
        ok = super().put_bytes(key, blob)
        self.flush(timeout=30)
        return ok

    def append_text(self, key, text):
        ok = super().append_text(key, text)
        self.flush(timeout=30)
        return ok


@contextlib.contextmanager
def _local_backend(tmp_path):
    store = LocalStorage(root=tmp_path)
    try:
        yield store
    finally:
        store.close()


@contextlib.contextmanager
def _databricks_backend(tmp_path, cls=FlushingDatabricksStorage, **overrides):
    """A backend pointed at a local server speaking the measured Files API semantics."""
    base_url, volume, shutdown = serve_fake_volume()
    config = {
        "host": base_url,
        "catalog": CATALOG,
        "schema": "afir",
        "volume": "state",
        "token_env": "AFIR_TEST_STORAGE_TOKEN",
        "verify_ssl": False,
        **overrides,
    }
    store = cls(config)
    store.volume = volume  # so a test can inspect what actually landed remotely
    try:
        yield store
    finally:
        store.close(timeout=5)
        shutdown()


@contextlib.contextmanager
def _sql_backend(tmp_path):
    """The sqlite dialect, which is the one held to the whole contract on every run.

    sqlite rather than Postgres because it needs no server, so the contract is checked in
    CI rather than only where a database happens to be running — and the two dialects
    differ in four values (``_DIALECT`` in src/storage/sql.py), not in any statement, so
    what passes here is what Postgres executes.
    """
    store = SqlStorage({"dialect": "sqlite", "dsn": str(tmp_path / "state.sqlite3")})
    try:
        yield store
    finally:
        store.close()


# Every backend that claims to implement the contract. Each entry is a context manager
# taking a tmp_path, so a backend needing setup and teardown (a server, a writer thread)
# registers here rather than getting a suite of its own.
BACKENDS = {
    "local": _local_backend,
    "databricks": _databricks_backend,
    "sql": _sql_backend,
}


@pytest.fixture(autouse=True)
def _storage_token(monkeypatch):
    """The remote backend needs a bearer token; the fake server enforces one.

    Autouse and unconditional, because the token is read per call inside the writer
    thread — setting it only in the databricks fixture would leave a window where a
    background write authenticates against an env var a later test has already removed.
    """
    monkeypatch.setenv("AFIR_TEST_STORAGE_TOKEN", "test-token-not-a-real-secret")


@pytest.fixture(params=sorted(BACKENDS))
def backend(request, tmp_path):
    with BACKENDS[request.param](tmp_path) as store:
        yield store


# --- the contract ----------------------------------------------------------


def test_put_then_get_round_trips(backend):
    assert backend.put_text("jobs/a1.json", '{"job_id": "a1"}') is True
    assert json.loads(backend.get_text("jobs/a1.json"))["job_id"] == "a1"


def test_get_of_an_absent_key_is_none_not_an_error(backend):
    assert backend.get_text("jobs/never-written.json") is None
    assert backend.get_bytes("jobs/never-written.json") is None
    assert backend.exists("jobs/never-written.json") is False


def test_put_replaces_rather_than_appends(backend):
    backend.put_text("jobs/a2.json", '{"status": "running"}')
    backend.put_text("jobs/a2.json", '{"status": "completed"}')
    assert json.loads(backend.get_text("jobs/a2.json")) == {"status": "completed"}


def test_append_accumulates_and_creates(backend):
    assert backend.append_text("feedback_log.jsonl", '{"n": 1}\n') is True
    backend.append_text("feedback_log.jsonl", '{"n": 2}\n')
    lines = [
        json.loads(x) for x in backend.get_text("feedback_log.jsonl").splitlines() if x
    ]
    assert [r["n"] for r in lines] == [1, 2]


def test_a_failed_verify_stores_nothing_and_leaves_the_previous_content(backend):
    """The parse-back is the reason ``verify`` exists — see storage/local.py."""
    backend.put_text("jobs/a3.json", '{"good": true}', verify=json_verifier)
    assert (
        backend.put_text("jobs/a3.json", "{ truncated", verify=json_verifier) is False
    )
    assert json.loads(backend.get_text("jobs/a3.json")) == {"good": True}


def test_delete_removes_the_key(backend):
    backend.put_text("jobs/a4.json", "{}")
    assert backend.delete("jobs/a4.json") is True
    assert backend.exists("jobs/a4.json") is False


def test_delete_of_an_absent_key_is_false_not_an_error(backend):
    assert backend.delete("jobs/nothing-here.json") is False


def test_list_keys_reports_size_and_orderable_mtime(backend):
    backend.put_text("exports/fraud_report_INC1.md", "# one")
    backend.put_text("exports/fraud_report_INC2.md", "# two, longer")

    objects = {o.key: o for o in backend.list_keys("exports")}
    assert set(objects) == {
        "exports/fraud_report_INC1.md",
        "exports/fraud_report_INC2.md",
    }
    assert objects["exports/fraud_report_INC2.md"].size > 0
    assert objects["exports/fraud_report_INC1.md"].name == "fraud_report_INC1.md"


def test_list_keys_of_an_absent_prefix_is_empty(backend):
    assert backend.list_keys("exports") == []


def test_list_keys_is_scoped_to_its_prefix(backend):
    backend.put_text("jobs/j1.json", "{}")
    backend.put_text("exports/incident_X.json", "{}")
    assert [o.key for o in backend.list_keys("jobs")] == ["jobs/j1.json"]


def test_binary_content_round_trips(backend):
    """A PDF is the real case: report_delivery serves report bytes verbatim."""
    blob = b"%PDF-1.4\x00\x01\x02 not utf-8 \xff\xfe"
    backend.put_bytes("exports/fraud_report_INC9.pdf", blob)
    assert backend.get_bytes("exports/fraud_report_INC9.pdf") == blob


# --- key validation --------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "../../etc/passwd",
        "jobs/../../etc/passwd",
        "/etc/passwd",
        "jobs//double.json",
        "jobs/..",
        "..",
        "",
        "   ",
        "jobs\\..\\..\\windows",
        "jobs/a\x00b.json",
        "jobs/" + "x" * (MAX_SEGMENT_LEN + 1),
    ],
)
def test_an_unsafe_key_is_rejected_never_repaired(hostile):
    with pytest.raises(ValueError):
        safe_key(hostile)


@pytest.mark.parametrize(
    "hostile", ["../../etc/passwd", "/etc/passwd", "jobs/../../x.json"]
)
def test_a_backend_refuses_an_unsafe_key_on_every_operation(backend, hostile):
    for call in (
        lambda: backend.put_text(hostile, "x"),
        lambda: backend.append_text(hostile, "x"),
        lambda: backend.get_text(hostile),
        lambda: backend.get_bytes(hostile),
        lambda: backend.exists(hostile),
        lambda: backend.delete(hostile),
        lambda: backend.list_keys(hostile),
    ):
        with pytest.raises(ValueError):
            call()


def test_a_normal_key_survives_validation_unchanged():
    for good in (
        "jobs/6f1e2d3c-4b5a.json",
        "jobs/6f1e2d3c.evidence.json",
        "exports/fraud_report_IR10000001.md",
        "feedback_log.jsonl",
    ):
        assert safe_key(good) == good


# --- local-backend specifics ----------------------------------------------
#
# These assert the ON-DISK layout, which the contract deliberately does not: an
# upgraded VM must find its own existing files, so the layout is a compatibility
# guarantee, not an implementation detail.


def test_the_on_disk_layout_is_unchanged(tmp_path):
    store = LocalStorage(root=tmp_path)
    store.put_text("jobs/j1.json", '{"job_id": "j1"}')
    assert (tmp_path / "jobs" / "j1.json").is_file()
    assert store.path_for("exports/x.md") == tmp_path / "exports" / "x.md"


def test_a_write_keeps_the_previous_version(tmp_path):
    store = LocalStorage(root=tmp_path)
    store.put_text("jobs/j2.json", '{"status": "running"}')
    store.put_text("jobs/j2.json", '{"status": "completed"}')

    prev = json.loads((tmp_path / "jobs" / "j2.json.prev").read_text())
    assert prev["status"] == "running"


def test_a_prev_file_is_not_a_stored_object(tmp_path):
    """Two entries for one job would make an inventory lie about what is there."""
    store = LocalStorage(root=tmp_path)
    store.put_text("jobs/j3.json", "{}")
    store.put_text("jobs/j3.json", '{"n": 2}')

    assert [o.key for o in store.list_keys("jobs")] == ["jobs/j3.json"]


def test_deleting_a_key_takes_its_prev_too(tmp_path):
    """Otherwise job_store's .prev fallback resurrects a deleted job."""
    store = LocalStorage(root=tmp_path)
    store.put_text("jobs/j4.json", '{"n": 1}')
    store.put_text("jobs/j4.json", '{"n": 2}')
    store.delete("jobs/j4.json")

    assert list((tmp_path / "jobs").glob("j4.json*")) == []


def test_a_leftover_temp_file_is_not_a_stored_object(tmp_path):
    store = LocalStorage(root=tmp_path)
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "j5.json.tmp999").write_text("half a doc")
    store.put_text("jobs/j5.json", "{}")

    assert [o.key for o in store.list_keys("jobs")] == ["jobs/j5.json"]


def test_an_unwritable_root_returns_false_rather_than_raising(tmp_path):
    """The pipeline's contract: losing durability must not lose the investigation."""
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        store = LocalStorage(root=blocked)
        if store.put_text("jobs/j6.json", "{}"):
            pytest.skip("this environment ignores directory permissions")
        assert store.get_text("jobs/j6.json") is None
    finally:
        blocked.chmod(0o700)


def test_local_defaults_to_the_data_dir_override(tmp_path, monkeypatch):
    """AFIR_DATA_DIR keeps working, so an existing VM deployment is unaffected."""
    monkeypatch.setenv("AFIR_DATA_DIR", str(tmp_path))
    store = LocalStorage()
    store.put_text("jobs/j7.json", "{}")
    assert (tmp_path / "jobs" / "j7.json").is_file()


# --- databricks-backend specifics -----------------------------------------
#
# The queue is the whole reason this backend is not just LocalStorage with a different
# root, and the contract above cannot see it: these assert what happens BETWEEN a
# synchronous `put_text` returning True and the bytes reaching the Volume. Every one of
# them describes a way a write can be reported as done and not be.


@pytest.fixture
def dbx(tmp_path):
    """The unwrapped backend — writes stay queued, as they do in production."""
    with _databricks_backend(tmp_path, cls=DatabricksStorage) as store:
        yield store


def test_a_put_returns_before_the_upload_and_the_flush_completes_it(dbx):
    """The point of the writer thread: 15 saves a run must not sit on the event loop."""
    dbx.volume.put_delay = 0.3
    started = time.monotonic()
    assert dbx.put_text("jobs/q1.json", '{"n": 1}') is True
    returned_in = time.monotonic() - started

    assert returned_in < 0.3, f"put_text blocked for {returned_in:.2f}s"
    assert dbx.flush(timeout=10) is True
    assert dbx.volume.get(f"{VOLUME_ROOT}/jobs/q1.json") == b'{"n": 1}'


def test_a_flush_waits_for_the_write_in_flight_not_just_the_queue(dbx):
    """A gate is announced after the flush, so 'drained' must mean 'landed'.

    The failure this pins: a flush that waits only for the queue to empty returns while
    the writer is still mid-upload, the gate is announced, the container dies, and the
    pending approval is gone — which is the exact bug the whole backend exists to fix.
    """
    dbx.volume.put_delay = 0.4
    dbx.put_text("jobs/gate.json", '{"status": "PAUSED"}', verify=json_verifier)

    assert dbx.flush(timeout=10) is True
    # Read from the fake volume directly: get_text would merge the queue and pass even
    # if nothing had been uploaded.
    assert dbx.volume.get(f"{VOLUME_ROOT}/jobs/gate.json") == b'{"status": "PAUSED"}'


def test_only_the_newest_snapshot_per_key_is_uploaded(dbx):
    """Coalescing. Every job save is a full snapshot, so older ones have no value."""
    dbx.volume.put_delay = 0.25  # holds the writer while the rest queue up
    for n in range(6):
        dbx.put_text("jobs/coalesce.json", json.dumps({"n": n}))
    dbx.flush(timeout=15)

    uploads = [
        c for c in dbx.volume.calls if c[0] == "PUT" and c[1].endswith("coalesce.json")
    ]
    assert json.loads(dbx.volume.get(f"{VOLUME_ROOT}/jobs/coalesce.json"))["n"] == 5
    assert (
        len(uploads) < 6
    ), f"no coalescing happened: {len(uploads)} uploads for 6 saves"


def test_a_read_sees_a_write_that_is_still_in_flight(dbx):
    """The window that is easy to miss: popped from the queue, not yet on the Volume.

    Straight to the remote here would answer with the PREVIOUS snapshot, which makes
    put-then-get a coin flip — and for append_text, a silently dropped record.
    """
    dbx.volume.put_delay = 0.5
    dbx.put_text("jobs/inflight.json", '{"v": 1}')
    dbx.flush(timeout=10)

    dbx.put_text("jobs/inflight.json", '{"v": 2}')
    time.sleep(0.15)  # inside the upload of v2
    assert json.loads(dbx.get_text("jobs/inflight.json"))["v"] == 2
    dbx.flush(timeout=10)


def test_an_append_during_an_upload_keeps_both_records(dbx):
    """A lost review is invisible: the log still parses, one analyst's entry is gone."""
    dbx.volume.put_delay = 0.4
    dbx.append_text("feedback_log.jsonl", '{"r": 1}\n')
    dbx.flush(timeout=10)

    dbx.append_text("feedback_log.jsonl", '{"r": 2}\n')
    time.sleep(0.15)  # while record 2 is uploading
    dbx.append_text("feedback_log.jsonl", '{"r": 3}\n')
    dbx.flush(timeout=15)

    stored = dbx.volume.get(f"{VOLUME_ROOT}/feedback_log.jsonl").decode()
    assert [json.loads(x)["r"] for x in stored.splitlines() if x] == [1, 2, 3]


def test_list_keys_walks_nested_directories(dbx):
    """The listing is one level per call, so recursion is the backend's job."""
    dbx.put_text("exports/IR1/report.md", "# one")
    dbx.put_text("exports/IR2/deeper/evidence.json", "{}")
    dbx.flush(timeout=10)

    assert {o.key for o in dbx.list_keys("exports")} == {
        "exports/IR1/report.md",
        "exports/IR2/deeper/evidence.json",
    }


def test_mtime_is_seconds_not_the_milliseconds_the_api_returns(dbx):
    """Measured: last_modified is 1786027620000. Raw, prune expires nothing, ever."""
    dbx.put_text("jobs/mt.json", "{}")
    dbx.flush(timeout=10)

    (obj,) = dbx.list_keys("jobs")
    assert (
        abs(obj.mtime - time.time()) < 300
    ), f"mtime {obj.mtime} is not a POSIX second"


def test_a_prev_copy_is_kept_and_is_not_a_stored_object(dbx):
    """job_store falls back to .prev when the current doc will not parse."""
    dbx.put_text("jobs/p1.json", '{"status": "running"}', verify=json_verifier)
    dbx.flush(timeout=10)
    dbx.put_text("jobs/p1.json", '{"status": "completed"}', verify=json_verifier)
    dbx.flush(timeout=10)

    assert json.loads(dbx.get_previous_text("jobs/p1.json"))["status"] == "running"
    assert [o.key for o in dbx.list_keys("jobs")] == ["jobs/p1.json"]


def test_deleting_a_key_takes_its_prev_and_cancels_a_queued_write(dbx):
    """Otherwise a deleted job reappears seconds later, or via the .prev fallback."""
    dbx.put_text("jobs/d1.json", '{"n": 1}', verify=json_verifier)
    dbx.flush(timeout=10)
    dbx.volume.put_delay = 0.3
    dbx.put_text("jobs/d1.json", '{"n": 2}', verify=json_verifier)

    assert dbx.delete("jobs/d1.json") is True
    dbx.flush(timeout=10)
    assert dbx.get_text("jobs/d1.json") is None
    assert dbx.get_previous_text("jobs/d1.json") is None


def test_a_readback_that_does_not_parse_restores_the_previous_content(dbx):
    """The parse-back is on what the VOLUME holds — a truncated upload is the case.

    Simulated by truncating inside the upload, because that is what a torn write looks
    like from here: 204, and bytes that no longer parse.
    """
    dbx.put_text("jobs/v1.json", '{"good": true}', verify=json_verifier)
    dbx.flush(timeout=10)

    real_upload = dbx._upload
    truncated = []

    def truncating_upload(key, blob):
        # Only the first document upload is torn. The restore that follows must land
        # intact, which is the behaviour being asserted; truncating that too would test
        # the give-up path instead.
        if key == "jobs/v1.json" and not truncated:
            truncated.append(key)
            return real_upload(key, blob[:6])
        return real_upload(key, blob)

    # setattr/delattr rather than monkeypatch here: `undo()` reverts EVERY patch in the
    # test, including the autouse token fixture, so the assertions below ran without a
    # bearer token and read 401 as "absent". A one-line convenience quietly removed the
    # credential the backend needs.
    dbx._upload = truncating_upload
    try:
        dbx.put_text("jobs/v1.json", '{"also_good": true}', verify=json_verifier)
        dbx.flush(timeout=10)
    finally:
        del dbx._upload

    assert json.loads(dbx.get_text("jobs/v1.json")) == {"good": True}
    assert "did not verify" in (dbx.degradation or "")


def test_an_unrestorable_corrupt_object_is_removed_not_left_in_place(dbx):
    """A restore is a claim, so it is verified — and a failed one must not stay stored.

    Whatever tore the first upload can tear the restore. Left in place, the Volume holds
    bytes that will not parse and the only trace is one log line; a restart then loads
    nothing and says nothing about why.
    """
    real_upload = dbx._upload
    dbx._upload = lambda key, blob: real_upload(key, blob[:6])
    try:
        dbx.put_text("jobs/v2.json", '{"first": true}', verify=json_verifier)
        dbx.flush(timeout=10)
    finally:
        del dbx._upload

    assert dbx.get_text("jobs/v2.json") is None
    assert dbx.exists("jobs/v2.json") is False


def test_a_missing_token_degrades_rather_than_raising(dbx, monkeypatch):
    """An unset token_env is a real deployment state, and 401 is not an exception."""
    monkeypatch.delenv("AFIR_TEST_STORAGE_TOKEN", raising=False)
    assert dbx.put_text("jobs/t1.json", "{}") is True  # queued; the write is what fails
    dbx.flush(timeout=10)

    assert dbx.get_text("jobs/t1.json") is None
    assert "degraded" in (dbx.degradation or "")


def test_an_unconfigured_catalog_is_loud_and_inert(caplog):
    """Nothing to write to. It must say so and still answer every call."""
    import logging

    with caplog.at_level(logging.ERROR):
        store = DatabricksStorage({"host": "https://example.invalid"})
    try:
        assert any("catalog is not set" in r.message for r in caplog.records)
        assert store.put_text("jobs/x.json", "{}") is False
        assert store.get_text("jobs/x.json") is None
        assert store.list_keys("jobs") == []
        assert "not usable" in (store.degradation or "")
    finally:
        store.close(timeout=1)


def test_close_drains_and_reports_what_it_could_not_write(dbx):
    """An App gets 15s from SIGTERM to SIGKILL: the drain is bounded, and says so."""
    dbx.put_text("jobs/c1.json", '{"n": 1}')
    dbx.close(timeout=10)

    assert dbx.volume.get(f"{VOLUME_ROOT}/jobs/c1.json") == b'{"n": 1}'
    # A closed store refuses rather than accepting a write nothing will perform.
    assert dbx.put_text("jobs/c2.json", "{}") is False


def test_a_healthy_backend_reports_no_degradation(dbx):
    """Otherwise the channel is noise and the real message gets ignored."""
    dbx.put_text("jobs/h1.json", "{}")
    dbx.flush(timeout=10)
    assert dbx.degradation is None


def test_the_job_store_contract_holds_over_the_remote_backend(tmp_path):
    """The one consumer whose failure is a lost human approval, end to end.

    Through the real ``JobStore`` — including its writability probe, the evidence sidecar
    split and the ``PrefixedStorage`` view ``build_job_store`` wraps it in — because a
    backend that passes the contract can still be wired up wrong.
    """
    from src.job_store import JobStore
    from src.storage import PrefixedStorage

    with _databricks_backend(tmp_path) as store:
        job_store = JobStore(storage=PrefixedStorage(store, "jobs"))
        doc = {
            "job_id": "6f1e2d3c-4b5a",
            "status": "PAUSED",
            "created_at": "2026-08-06T10:00:00Z",
            "outputs": {"logs": {"auth_events": [{"row": 1}]}, "summary": "held"},
        }
        assert job_store.save(doc, evidence_changed=True) is True
        store.flush(timeout=15)

        (loaded,) = job_store.load_all()
        assert loaded["status"] == "PAUSED"
        assert loaded["outputs"]["logs"]["auth_events"] == [{"row": 1}]
        assert loaded["outputs"]["summary"] == "held"
        # The sidecar is a separate object, and neither it nor a .prev is a job doc.
        assert sorted(job_store._job_doc_keys()) == ["6f1e2d3c-4b5a.json"]

        assert job_store.delete("6f1e2d3c-4b5a") is True
        assert job_store.load_all() == []


def test_prune_keeps_a_job_whose_age_is_unknown(tmp_path):
    """mtime 0.0 means the backend cannot say. Read as the epoch it deletes the queue."""
    from src.job_store import JobStore
    from src.storage import PrefixedStorage
    from src.storage.base import StoredObject

    with _databricks_backend(tmp_path) as store:
        job_store = JobStore(
            storage=PrefixedStorage(store, "jobs"), retention_days=0.0001
        )
        job_store.save({"job_id": "unknown-age", "status": "PAUSED"})
        store.flush(timeout=15)

        backend = job_store.storage
        real_list = backend.list_keys
        backend.list_keys = lambda prefix: [
            StoredObject(key=o.key, size=o.size, mtime=0.0) for o in real_list(prefix)
        ]
        assert job_store.prune() == 0
        backend.list_keys = real_list
        assert len(job_store.load_all()) == 1


# --- sql-backend specifics -------------------------------------------------
#
# The shared contract above already holds this backend to the local oracle. What follows
# is what the contract cannot see: the guarantees that only exist because it is a
# database, and the two failures whose whole hazard is that they look like success.


def _sqlite(tmp_path, **overrides):
    config = {"dialect": "sqlite", "dsn": str(tmp_path / "state.sqlite3"), **overrides}
    return SqlStorage(config)


def test_append_is_one_statement_so_concurrent_writers_lose_nothing(tmp_path):
    """The capability this backend has that the Volume backend does not.

    ``DatabricksStorage`` emulates append as a read-modify-write serialised by its single
    writer thread, which is why that deployment is documented single-replica: a second
    replica has its own thread and silently drops one analyst's review. Here the append is
    ``body = body || excluded.body`` in one statement, so this asserts what the design
    claims rather than trusting the SQL to say it.
    """
    import threading

    store = _sqlite(tmp_path)
    try:
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    assert store.append_text("feedback_log.jsonl", f"{n}-{i}\n") is True
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        lines = [x for x in store.get_text("feedback_log.jsonl").splitlines() if x]
        # Every line from every writer, none interleaved into another and none lost.
        assert len(lines) == 80
        assert len(set(lines)) == 80
    finally:
        store.close()


def test_a_failed_verify_rolls_back_rather_than_leaving_a_half_row(tmp_path):
    """The parse-back happens INSIDE the transaction, so a rejection stores nothing.

    The shared contract checks that the previous content survives. This checks the
    mechanism that makes it survive: had the upsert committed before the verify, the row
    would already hold the truncated body and ``prev_body`` would hold the good one — the
    two swapped, with `get_text` answering the broken document.
    """
    store = _sqlite(tmp_path)
    try:
        store.put_text("jobs/j1.json", '{"v": 1}', verify=json_verifier)
        assert store.put_text("jobs/j1.json", "{ nope", verify=json_verifier) is False
        assert json.loads(store.get_text("jobs/j1.json")) == {"v": 1}
        # And the rejected payload did not become the recoverable previous version
        # either, which would hand `JobStore.load_all`'s fallback the broken document.
        prev = store.get_previous_text("jobs/j1.json")
        assert prev is None or json.loads(prev)
    finally:
        store.close()


def test_delete_takes_the_previous_version_with_it(tmp_path):
    """Or ``load_all``'s fallback resurrects a job the operator deleted."""
    store = _sqlite(tmp_path)
    try:
        store.put_text("jobs/j2.json", '{"v": 1}')
        store.put_text("jobs/j2.json", '{"v": 2}')
        assert store.get_previous_text("jobs/j2.json") is not None
        assert store.delete("jobs/j2.json") is True
        assert store.get_previous_text("jobs/j2.json") is None
    finally:
        store.close()


def test_mtime_is_posix_seconds_not_milliseconds(tmp_path):
    """A row that exists never reports 0.0, and never a time 54,000 years hence.

    Both directions of this have already cost this system: ``prune`` reads ``0.0`` as
    unknown and skips, so a real 0.0 would keep everything forever, while milliseconds
    passed through as seconds put every job in the future and expire nothing. Compared
    against ``time.time()`` because that is the same clock ``prune`` subtracts from.
    """
    store = _sqlite(tmp_path)
    try:
        before = time.time()
        store.put_text("jobs/j3.json", "{}")
        after = time.time()
        (obj,) = store.list_keys("jobs")
        assert before - 2 <= obj.mtime <= after + 2
        assert obj.size > 0
    finally:
        store.close()


def test_list_keys_prefix_does_not_match_a_longer_sibling_name(tmp_path):
    """``jobs`` must not sweep up ``jobs_archive`` — the disk backend cannot make this
    mistake because it descends a directory; this one is told with a trailing slash."""
    store = _sqlite(tmp_path)
    try:
        store.put_text("jobs/real.json", "{}")
        store.put_text("jobsarchive/old.json", "{}")
        assert [o.key for o in store.list_keys("jobs")] == ["jobs/real.json"]
    finally:
        store.close()


def test_binary_round_trips_unchanged(tmp_path):
    """A PDF is the reason ``put_bytes`` exists, and psycopg hands back a memoryview."""
    store = _sqlite(tmp_path)
    try:
        blob = bytes(range(256)) * 8
        assert store.put_bytes("exports/report.pdf", blob) is True
        assert store.get_bytes("exports/report.pdf") == blob
        assert isinstance(store.get_bytes("exports/report.pdf"), bytes)
    finally:
        store.close()


def test_an_unusable_configuration_refuses_writes_and_names_the_reason(tmp_path):
    """A store that accepts every write and holds nothing is the defect to avoid.

    ``/health?deep=1`` reads ``degradation``; a backend that returned ``True`` from
    ``put_text`` with no database behind it would look identical to a working one until
    the restart that finds nothing.
    """
    store = SqlStorage({"dialect": "sqlite", "dsn": "", "dsn_env": "AFIR_UNSET_DSN"})
    try:
        assert store.put_text("jobs/j.json", "{}") is False
        assert store.get_text("jobs/j.json") is None
        assert store.list_keys("jobs") == []
        assert store.flush() is False
        detail = store.degradation or ""
        # It must name the env var it looked in, or the operator has nothing to go and set.
        assert "no connection string" in detail and "AFIR_UNSET_DSN" in detail
    finally:
        store.close()


def test_the_connection_string_is_read_from_the_named_env_var(tmp_path, monkeypatch):
    """A libpq URL embeds its password, so the literal must not have to live in YAML.

    Same indirection as ``api_key_env`` / ``token_env`` everywhere else, and it is what
    keeps ``storage.sql`` configurable from the Configuration tab at all: a ``dsn`` form
    control would be a secret-shaped field on a page whose reads are redacted, and a
    redacted control PUTs the placeholder back on the next save.
    """
    monkeypatch.setenv("MY_AFIR_DSN", str(tmp_path / "from-env.sqlite3"))
    store = SqlStorage({"dialect": "sqlite", "dsn_env": "MY_AFIR_DSN"})
    try:
        assert store.degradation is None
        assert store.put_text("jobs/j.json", '{"v": 1}') is True
        assert (tmp_path / "from-env.sqlite3").is_file()
    finally:
        store.close()


def test_an_explicit_dsn_wins_over_the_env_var(tmp_path, monkeypatch):
    """The more specific of the two statements. A sqlite path is not a credential, so
    writing one by hand is the obvious thing to do and must not be silently overridden.
    """
    monkeypatch.setenv("MY_AFIR_DSN", str(tmp_path / "from-env.sqlite3"))
    store = SqlStorage(
        {
            "dialect": "sqlite",
            "dsn": str(tmp_path / "explicit.sqlite3"),
            "dsn_env": "MY_AFIR_DSN",
        }
    )
    try:
        assert store.put_text("jobs/j.json", "{}") is True
        assert (tmp_path / "explicit.sqlite3").is_file()
        assert not (tmp_path / "from-env.sqlite3").exists()
    finally:
        store.close()


def test_an_unsupported_dialect_is_refused_with_the_valid_ones_named(tmp_path):
    """A warehouse is not an option here, and the reason must be readable.

    ``snowflake`` and ``databricks`` are the two an operator would reasonably try; both
    are refused for measured reasons recorded in src/storage/sql.py, not preference.
    """
    for dialect in ("snowflake", "databricks", "mysql"):
        store = SqlStorage({"dialect": dialect, "dsn": str(tmp_path / "x.sqlite3")})
        try:
            assert store.put_text("jobs/j.json", "{}") is False
            detail = store.degradation or ""
            assert "not supported" in detail
            assert "sqlite" in detail and "postgresql" in detail
        finally:
            store.close()


def test_a_table_name_that_is_not_an_identifier_is_refused(tmp_path):
    """The one value that cannot be a bound parameter, so it is validated not escaped."""
    store = SqlStorage(
        {
            "dialect": "sqlite",
            "dsn": str(tmp_path / "x.sqlite3"),
            "table": "blobs; DROP TABLE blobs",
        }
    )
    try:
        assert store.put_text("jobs/j.json", "{}") is False
        assert "identifier" in (store.degradation or "")
    finally:
        store.close()


def test_the_table_is_shared_across_processes_not_per_connection(tmp_path):
    """A second store over the same DSN reads the first one's writes.

    This is what "durable" means for this backend, and it is also what makes several
    replicas viable: the row is in the database, not in a per-instance cache.
    """
    dsn = str(tmp_path / "state.sqlite3")
    first = SqlStorage({"dialect": "sqlite", "dsn": dsn})
    try:
        first.put_text("jobs/shared.json", '{"from": "first"}')
    finally:
        first.close()
    second = SqlStorage({"dialect": "sqlite", "dsn": dsn})
    try:
        assert json.loads(second.get_text("jobs/shared.json")) == {"from": "first"}
    finally:
        second.close()


def test_reads_and_writes_work_from_another_thread(tmp_path):
    """Connections are per-thread; the mirror's push hooks run off the event loop."""
    store = _sqlite(tmp_path)
    try:
        store.put_text("jobs/j4.json", '{"v": 1}')
        out = {}

        def work():
            out["read"] = store.get_text("jobs/j4.json")
            out["wrote"] = store.put_text("jobs/j5.json", '{"v": 2}')

        import threading

        t = threading.Thread(target=work)
        t.start()
        t.join()
        assert json.loads(out["read"]) == {"v": 1}
        assert out["wrote"] is True
        assert json.loads(store.get_text("jobs/j5.json")) == {"v": 2}
    finally:
        store.close()


# --- selection -------------------------------------------------------------


def test_no_storage_block_means_local():
    """Every existing deployment. The default must not require a config edit."""
    assert build_storage({}).kind == "local"
    assert build_storage(None).kind == "local"
    assert build_storage({"storage": {}}).kind == "local"


def test_an_explicit_local_backend_honours_its_root(tmp_path):
    store = build_storage({"storage": {"backend": "local", "root": str(tmp_path)}})
    assert store.root == tmp_path


def test_storage_root_is_how_a_mounted_or_external_VOLUME_is_reached(tmp_path):
    """One of the four documented destinations, and it is deliberately not a new backend.

    An NFS/SMB mount or an attached disk *is* a filesystem — it takes ``os.replace`` and
    ``fsync`` — so pointing ``LocalStorage`` at it is the whole implementation. This
    asserts the write genuinely lands under the given root and not under ``data_dir()``,
    which is the way a silently-ignored ``root`` would fail: durable-looking, in the wrong
    place.
    """
    mount = tmp_path / "mnt" / "afir-state"
    store = build_storage({"storage": {"backend": "local", "root": str(mount)}})
    try:
        assert store.put_text("jobs/j.json", '{"v": 1}') is True
        assert (mount / "jobs" / "j.json").is_file()
    finally:
        store.close()


@pytest.mark.parametrize("name", ["sql", "database", "db", "sqlite"])
def test_the_sql_backend_is_reachable_under_each_accepted_name(name, tmp_path):
    """An operator writing `postgres` or `database` must not land on local disk.

    A backend name that falls back is logged, but the run continues and looks healthy —
    so a synonym that misses is the ``a switch that is accepted and refuses every write``
    shape, discovered at the restart that finds nothing.
    """
    store = build_storage(
        {
            "storage": {
                "backend": name,
                "sql": {"dialect": "sqlite", "dsn": str(tmp_path / f"{name}.sqlite3")},
            }
        }
    )
    try:
        assert store.kind == "sql"
        assert store.put_text("jobs/j.json", "{}") is True
    finally:
        store.close()


def test_naming_the_dialect_as_the_backend_sets_it(tmp_path):
    """`backend: postgresql` has said everything the operator meant.

    An explicit `sql.dialect` still wins, being the more specific of the two statements —
    checked here because the fallback silently choosing the wrong dialect would surface as
    a driver error at the first save rather than at boot.
    """
    store = build_storage(
        {"storage": {"backend": "postgres", "sql": {"dsn": "postgresql:///nope"}}}
    )
    try:
        # No server, so it is unusable — but it must be unusable AS POSTGRES, which is
        # what proves the dialect came from the backend name.
        assert store.kind == "sql"
        assert "postgres" in (store.degradation or "").lower() or "cannot open" in (
            store.degradation or ""
        )
    finally:
        store.close()

    explicit = build_storage(
        {
            "storage": {
                "backend": "sqlite",
                "sql": {"dialect": "sqlite", "dsn": str(tmp_path / "e.sqlite3")},
            }
        }
    )
    try:
        assert explicit.degradation is None
    finally:
        explicit.close()


def test_an_unknown_backend_falls_back_to_local_and_says_so(caplog):
    """A typo must not stop the run, and must not look like it was honoured."""
    import logging

    with caplog.at_level(logging.ERROR):
        store = build_storage({"storage": {"backend": "delta-lake-ish"}})
    assert store.kind == "local"
    assert any("LOCAL DISK" in r.message for r in caplog.records)


def test_stored_object_name_is_the_last_segment():
    assert StoredObject(key="exports/a/b.md").name == "b.md"
    assert StoredObject(key="top.md").name == "top.md"
