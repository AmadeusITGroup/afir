"""One caller's own credential in place of the deployment's — the store's own semantics.

`test_identity_api.py` proves the three routes reach this module; what only a unit test can
ask is the set of properties that each fail *silently* when they stop holding. Every test
here is one of them:

- a name no reader resolves is refused, rather than stored to no effect;
- the value never comes back out of a read-back surface, only a fingerprint;
- one caller's credential is not another's, and is not the deployment's;
- and with no store installed and no caller bound, every reader is byte-identical to the tree
  that shipped before this file existed — which is what keeps a laptop, a VM and an App
  Service on the path they were on.

The store is `LocalStorage` throughout, because it is the oracle `test_storage.py`
parametrises every backend against: a property asserted against a dict double would be a
property of the double.
"""

import json
import os

import pytest

from src import user_secrets
from src.storage import LocalStorage
from src.user_secrets import (CACHE_TTL_SECONDS, MAX_SECRET_CHARS,
                              SecretsUnavailable, UserSecretStore, fingerprint,
                              offered_names, personal_value,
                              reset_current_segment, resolve_env, secret_store,
                              set_current_segment, set_secret_store,
                              withheld_names)

NAME = "AFIR_TEST_TOKEN"
OTHER = "AFIR_TEST_SECOND"
OFFERED = {NAME: ["LLM reasoning (every stage)"], OTHER: ["a REST endpoint"]}

ONE = "someone@example.test"
TWO = "somebody-else@example.test"


def row_for(store, segment, name):
    """The row `describe` shows for one name. By name, because it sorts its whole set."""
    (row,) = [r for r in store.describe(segment) if r["name"] == name]
    return row


@pytest.fixture
def store(tmp_path):
    return UserSecretStore(LocalStorage(root=tmp_path / "store"), offered=OFFERED)


@pytest.fixture(autouse=True)
def no_installed_store():
    """The module-level store is process-global, so a leak lands in whichever test runs next."""
    set_secret_store(None)
    yield
    set_secret_store(None)


@pytest.fixture
def bound():
    """A caller bound the way the HTTP seam and a detached run bind one, then unbound."""

    def _bind(segment):
        return set_current_segment(segment)

    tokens = []
    yield lambda segment: tokens.append(_bind(segment))
    for token in reversed(tokens):
        reset_current_segment(token)


# -- what may be stored at all ------------------------------------------------


def test_a_name_no_reader_resolves_is_refused_and_nothing_is_written(store, tmp_path):
    """The silent no-op this codebase forbids: a stored secret, a 200, and no run changed."""
    with pytest.raises(KeyError):
        store.set(ONE, "SOMETHING_NOBODY_READS", "s3cret")
    assert store.resolve(ONE, "SOMETHING_NOBODY_READS") is None
    assert not list((tmp_path / "store").rglob("*.json"))


def test_a_name_that_stopped_being_read_stops_being_honoured(tmp_path):
    """The offered set is recomputed from the config at boot, so a name dropped from the
    config must stop applying rather than apply to nothing — the stored value is still there
    and must not be resolved."""
    backend = LocalStorage(root=tmp_path / "store")
    UserSecretStore(backend, offered=OFFERED).set(ONE, NAME, "v1")
    after = UserSecretStore(backend, offered={OTHER: ["a REST endpoint"]})
    assert after.resolve(ONE, NAME) is None
    assert [row["name"] for row in after.describe(ONE)] == [OTHER]


def test_the_two_refusals_cannot_be_caught_by_one_clause(store):
    """"Nowhere to keep it" and "nothing reads that name" are both the store saying no, and they
    need opposite answers: 503 come back later, against 400 you asked for the wrong thing.
    `KeyError` IS a `LookupError`, so a handler catching the wider one first answered every
    unreadable name with a 503 holding the repr of that name — and the refusal that lists the
    names it *does* read was unreachable."""
    with pytest.raises(SecretsUnavailable):
        UserSecretStore(None, offered=OFFERED).set(ONE, NAME, "v1")
    with pytest.raises(KeyError) as caught:
        store.set(ONE, "SOMETHING_NOBODY_READS", "v1")
    assert not isinstance(caught.value, SecretsUnavailable)
    assert not issubclass(SecretsUnavailable, LookupError)


def test_an_empty_value_is_refused_because_DELETE_is_the_way_back(store):
    with pytest.raises(ValueError):
        store.set(ONE, NAME, "   ")
    assert store.resolve(ONE, NAME) is None


def test_a_mispaste_of_a_whole_file_is_refused_rather_than_stored(store):
    with pytest.raises(ValueError):
        store.set(ONE, NAME, "x" * (MAX_SECRET_CHARS + 1))
    assert store.resolve(ONE, NAME) is None


def test_a_write_that_does_not_land_is_an_error_and_not_a_silent_loss(tmp_path):
    """A refused write reported as success reads, at the next run, as "the caller never set
    one" — the two are indistinguishable from the run's side, so the failure has to surface
    here."""

    class Refusing(LocalStorage):
        def put_text(self, *a, **k):
            super().put_text(*a, **k)
            return False

    store = UserSecretStore(Refusing(root=tmp_path / "store"), offered=OFFERED)
    with pytest.raises(OSError):
        store.set(ONE, NAME, "v1")


# -- the value never comes back ----------------------------------------------


def test_no_read_back_surface_carries_the_value(store):
    """`describe` and the row a write returns are the whole read surface. Asserted over every
    value in the row and not only over a `value` key, because the leak that matters is the
    secret appearing under any name at all."""
    row = store.set(ONE, NAME, "sup3r-s3cret")
    for shape in (row, row_for(store, ONE, NAME)):
        assert "sup3r-s3cret" not in json.dumps(shape)
        assert shape["fingerprint"] == fingerprint("sup3r-s3cret")
        assert shape["personal"] is True and shape["source"] == "personal"


def test_a_fingerprint_confirms_a_paste_and_tells_two_credentials_apart(store):
    first = store.set(ONE, NAME, "v1")["fingerprint"]
    second = store.set(ONE, NAME, "v2")["fingerprint"]
    assert first and second and first != second
    assert len(first) < len("v1sup3r") + 64  # a prefix of the digest, not the digest


def test_the_only_reader_is_resolve_and_it_needs_the_name_and_the_caller(store):
    store.set(ONE, NAME, "v1")
    assert store.resolve(ONE, NAME) == "v1"
    assert store.resolve(ONE, OTHER) is None
    assert store.resolve("", NAME) is None


# -- one caller is not another ------------------------------------------------


def test_one_callers_credential_is_invisible_to_every_other_caller(store):
    store.set(ONE, NAME, "mine")
    assert store.resolve(TWO, NAME) is None
    assert row_for(store, TWO, NAME)["personal"] is False
    assert row_for(store, TWO, NAME)["fingerprint"] == ""


def test_two_callers_hold_two_values_for_the_same_name(store):
    store.set(ONE, NAME, "mine")
    store.set(TWO, NAME, "theirs")
    assert store.resolve(ONE, NAME) == "mine"
    assert store.resolve(TWO, NAME) == "theirs"


def test_the_file_sits_beside_the_per_caller_namespace_and_not_inside_jobs(store):
    """`JobStore.prune()` deletes any `.json` in its own namespace past the retention window,
    whether or not it parses as a job document — so a credential filed under `jobs/` would
    quietly expire and the run would read as one where nobody had set anything."""
    key = store._key(ONE)
    assert key.startswith("users/") and key.endswith("/credentials.json")
    assert "/jobs/" not in key


def test_a_clear_goes_back_to_the_deployments_value_and_says_so(store):
    store.set(ONE, NAME, "mine")
    row = store.clear(ONE, NAME)
    assert row["personal"] is False and row["source"] == "shared"
    assert row["fingerprint"] == ""
    assert store.resolve(ONE, NAME) is None


def test_clearing_what_was_never_set_is_a_different_answer_from_an_unknown_name(store):
    """Both are `KeyError` from here, which is why the handler checks `offered` first — but a
    caller's own empty state must not read as a deployment that cannot offer the name."""
    with pytest.raises(KeyError):
        store.clear(ONE, NAME)
    assert NAME in store.offered


# -- reading is best-effort ---------------------------------------------------


def test_an_unreadable_credential_file_never_fails_a_run(tmp_path, caplog):
    class Broken(LocalStorage):
        def get_text(self, key):
            raise OSError("the volume is gone")

    store = UserSecretStore(Broken(root=tmp_path / "store"), offered=OFFERED)
    assert store.resolve(ONE, NAME) is None
    assert [row["personal"] for row in store.describe(ONE)] == [False, False]


def test_credentials_that_are_not_json_read_as_no_credentials(tmp_path):
    backend = LocalStorage(root=tmp_path / "store")
    store = UserSecretStore(backend, offered=OFFERED)
    backend.put_text(store._key(ONE), "not json at all")
    assert store.resolve(ONE, NAME) is None


def test_a_second_replicas_write_is_picked_up_because_the_cache_is_BOUNDED(
    tmp_path, monkeypatch
):
    """`SqlStorage` genuinely supports a second replica, so a cache held forever is a caller
    whose own save never applies. Bounded, and short enough that their next run sees it."""
    backend = LocalStorage(root=tmp_path / "store")
    store = UserSecretStore(backend, offered=OFFERED)
    assert store.resolve(ONE, NAME) is None          # loads and caches the empty set
    UserSecretStore(backend, offered=OFFERED).set(ONE, NAME, "written elsewhere")
    assert store.resolve(ONE, NAME) is None          # still inside the window

    real = user_secrets.time.time
    monkeypatch.setattr(user_secrets.time, "time",
                        lambda: real() + CACHE_TTL_SECONDS + 1)
    assert store.resolve(ONE, NAME) == "written elsewhere"


# -- who is asking, out of band -----------------------------------------------


def test_with_no_store_installed_resolve_env_IS_os_environ_get(monkeypatch):
    """The property that keeps a laptop, a VM and an App Service on the path they were on:
    every reader calls this seam, and off a per-caller deployment it is the standard library."""
    monkeypatch.setenv(NAME, "the deployments own")
    assert secret_store() is None
    assert personal_value(NAME) is None
    assert resolve_env(NAME) == os.environ.get(NAME) == "the deployments own"
    monkeypatch.delenv(NAME)
    assert resolve_env(NAME) is None and os.environ.get(NAME) is None


def test_with_a_store_but_no_caller_bound_nothing_personal_resolves(store, monkeypatch):
    """A boot-time reader — the RAG index builder — runs before any caller exists, and must
    get the deployment's credential rather than an arbitrary one."""
    monkeypatch.setenv(NAME, "the deployments own")
    store.set(ONE, NAME, "mine")
    set_secret_store(store)
    assert personal_value(NAME) is None
    assert resolve_env(NAME) == "the deployments own"


def test_a_bound_caller_gets_their_own_and_everyone_else_the_deployments(
    store, bound, monkeypatch
):
    monkeypatch.setenv(NAME, "the deployments own")
    store.set(ONE, NAME, "mine")
    set_secret_store(store)
    bound(ONE)
    assert personal_value(NAME) == "mine"
    assert resolve_env(NAME) == "mine"
    bound(TWO)
    assert personal_value(NAME) is None
    assert resolve_env(NAME) == "the deployments own"


def test_a_bound_caller_with_nothing_of_their_own_falls_back(store, bound, monkeypatch):
    monkeypatch.setenv(NAME, "the deployments own")
    set_secret_store(store)
    bound(ONE)
    assert resolve_env(NAME) == "the deployments own"


def test_an_unbind_restores_the_deployments_credential(store, monkeypatch):
    """The HTTP seam resets in a `finally`, so a request that raised must not leave the next
    one — or a boot-time reader on the same loop — resolving somebody's personal token."""
    monkeypatch.setenv(NAME, "the deployments own")
    store.set(ONE, NAME, "mine")
    set_secret_store(store)
    token = set_current_segment(ONE)
    assert personal_value(NAME) == "mine"
    reset_current_segment(token)
    assert personal_value(NAME) is None
    assert resolve_env(NAME) == "the deployments own"


def test_a_detached_run_inherits_the_binding_it_was_created_under(store, monkeypatch):
    """A task created inside the bound window gets a *copy* of the context, which is why the
    request's own `finally` can reset while the run it started keeps the caller."""
    import asyncio

    monkeypatch.setenv(NAME, "the deployments own")
    store.set(ONE, NAME, "mine")
    set_secret_store(store)

    async def scenario():
        seen = []

        async def run():
            await asyncio.sleep(0)
            seen.append(personal_value(NAME))

        token = set_current_segment(ONE)
        task = asyncio.ensure_future(run())
        reset_current_segment(token)
        await task
        return seen

    assert asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        scenario()
    ) == ["mine"]


# -- which names may be offered ----------------------------------------------


def test_one_name_backing_several_subsystems_is_offered_ONCE_naming_all_of_them(store):
    """The reason the unit is the environment-variable name and not the subsystem: asked per
    subsystem, the same secret is requested four times and three of the four go stale."""
    offered = offered_names(
        main_config={
            "rag": {"embedding_token_env": "SHARED_TOKEN"},
            "log_sources": {
                "backends": {
                    "databricks": {"analytics": {"api_key_env": "SHARED_TOKEN"}},
                    "rest": {"tickets": {"token_env": "TICKET_TOKEN"}},
                }
            },
        },
        llm_config={"api_key_env": "SHARED_TOKEN"},
    )
    assert sorted(offered) == ["SHARED_TOKEN", "TICKET_TOKEN"]
    assert len(offered["SHARED_TOKEN"]) == 3
    assert any("embedding" in s.lower() for s in offered["SHARED_TOKEN"])
    assert any("analytics" in s for s in offered["SHARED_TOKEN"])


def test_a_deployment_reading_no_named_credential_offers_nothing(tmp_path):
    store = UserSecretStore(LocalStorage(root=tmp_path / "store"), offered={})
    assert offered_names({}, {}) == {}
    assert store.available is False


def test_the_two_reasons_the_feature_is_off_need_different_fixes(tmp_path):
    no_store = UserSecretStore(None, offered=OFFERED)
    assert no_store.available is False
    assert "nowhere" in no_store.unavailable_reason

    nothing_read = UserSecretStore(LocalStorage(root=tmp_path / "s"), offered={})
    assert "no credential by name" in nothing_read.unavailable_reason

    working = UserSecretStore(LocalStorage(root=tmp_path / "s"), offered=OFFERED)
    assert working.available is True and working.unavailable_reason == ""


def test_the_withheld_names_are_reported_rather_than_omitted():
    """A surface listing four names and silently dropping three others reads as complete."""
    withheld = withheld_names(
        {
            "storage": {"databricks": {"token_env": "STORE_TOKEN"}},
            "log_sources": {"backends": {"elasticsearch": {"main": {"hosts": []}}}},
        }
    )
    names = [row["name"] for row in withheld]
    assert "STORE_TOKEN" in names
    assert any("Snowflake" in n or "Elasticsearch" in n for n in names)
    assert all(row["reason"] for row in withheld)


def test_a_deployment_with_no_shared_state_and_no_sign_in_withholds_nothing():
    assert withheld_names({}) == []


def test_a_shared_state_name_a_caller_CAN_replace_is_named_for_the_subsystem():
    """One token backs the reasoning endpoint and the store on this deployment, so the flat row
    put the only replaceable name under "Not replaceable", saying a paste of it "would do
    nothing" — advice against the one thing the feature exists for. The limit is real and it is
    durable state's, so that is what the row names; the credential is still offered."""
    cfg = {"storage": {"backend": "dbfs", "databricks": {"token_env": NAME}}}
    (row,) = withheld_names(cfg, offered={NAME: ["LLM reasoning (every stage)"]})
    assert row["name"] != NAME and NAME in row["reason"]
    assert "for your runs" in row["reason"]
    (flat,) = withheld_names(cfg)
    assert flat["name"] == NAME
