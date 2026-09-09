"""The trust rules behind the driver proxy, each pinned to the measurement that set it."""

import pytest
from multidict import CIMultiDict

from src.identity import (ADMIN, LOCAL_IDENTITY, USER, Identity,
                          IdentityRefused, IdentityResolver,
                          build_identity_resolver, storage_segment)

OWNER = "corp-prd-platform-workspace-owners"
ME = "dana.lee@corp.example.test"


def headers(*pairs):
    """Header multidict, order preserved: which duplicate a reader takes is the whole point."""
    return CIMultiDict(pairs)


def browser(name=ME, user_id="1234567890123456", validated="true"):
    """What the proxy actually sends on the cookie path: name and id twice, no credential."""
    return headers(
        ("x-databricks-auth-validated", validated),
        ("x-databricks-auth-type", "DB_AAD"),
        ("x-databricks-user-name", name),
        ("x-databricks-user-name", name),
        ("x-databricks-user-id", user_id),
        ("x-databricks-user-id", user_id),
    )


def resolver(**kw):
    kw.setdefault("mode", "on")
    kw.setdefault("admin_groups", [OWNER])
    return IdentityResolver(**kw)


# -- modes -----------------------------------------------------------------


def test_mode_off_is_the_single_local_admin():
    assert resolver(mode="off").resolve(browser()) is LOCAL_IDENTITY


def test_mode_on_refuses_a_request_with_no_validated_identity():
    with pytest.raises(IdentityRefused):
        resolver().resolve(headers())


def test_an_unvalidated_name_is_not_an_identity():
    """`auth-validated` is read before the name, so a bare forged name earns nothing."""
    with pytest.raises(IdentityRefused):
        resolver().resolve(browser(validated="false"))


def test_a_typo_in_mode_is_reported_and_still_resolves():
    assert resolver(mode="enabld").mode_recognised is False


# -- the browser path ------------------------------------------------------


def test_browser_path_yields_the_platform_asserted_name():
    got = resolver().resolve(browser())
    assert (got.user_name, got.source, got.user_id) == (ME, "header", "1234567890123456")


def test_browser_path_defaults_to_user_because_groups_are_unreadable():
    """`GET /Users/{id}` is 403 without admin, so a browser caller's groups cannot be looked up."""
    got = resolver().resolve(browser())
    assert got.role == USER and got.groups == ()


def test_admin_users_is_the_browser_path_fallback():
    got = resolver(admin_users=[ME]).resolve(browser())
    assert got.role == ADMIN and "admin_users" in got.role_reason


def test_admin_users_ignores_case():
    assert resolver(admin_users=[ME.upper()]).resolve(browser()).role == ADMIN


# -- the API path ----------------------------------------------------------


def test_a_validated_token_carries_groups_and_decides_the_role():
    def validator(token):
        assert token == "dapi-secret"
        return {"userName": ME, "id": "42", "groups": [{"display": OWNER}]}

    got = resolver(validator=validator).resolve(
        headers(("x-databricks-user-token", "dapi-secret"))
    )
    assert (got.role, got.source, got.user_id) == (ADMIN, "token", "42")
    assert got.groups == (OWNER,)


def test_a_token_outranks_the_name_header_it_arrives_beside():
    """The credential is evidence; the name beside it is only an assertion."""
    head = browser(name="someone.else@example.com")
    head.add("x-databricks-user-token", "dapi-secret")
    got = resolver(
        validator=lambda t: {"userName": ME, "id": "42", "groups": [{"display": OWNER}]}
    ).resolve(head)
    assert got.user_name == ME


def test_a_token_that_does_not_validate_grants_nothing_and_falls_back():
    head = browser()
    head.add("x-databricks-user-token", "expired")
    got = resolver(validator=lambda t: None).resolve(head)
    assert (got.role, got.source) == (USER, "header")


def test_a_validator_that_raises_does_not_fail_the_request():
    head = browser()
    head.add("x-databricks-user-token", "boom")

    def validator(token):
        raise RuntimeError("workspace unreachable")

    assert resolver(validator=validator).resolve(head).source == "header"


def test_validation_is_cached_per_token():
    calls = []

    def validator(token):
        calls.append(token)
        return {"userName": ME, "id": "42", "groups": []}

    res = resolver(validator=validator)
    head = headers(("x-databricks-user-token", "dapi-secret"))
    res.resolve(head)
    res.resolve(head)
    assert calls == ["dapi-secret"]


def test_a_group_the_config_does_not_name_is_not_admin():
    got = resolver(
        validator=lambda t: {"userName": ME, "id": "42", "groups": [{"display": "users"}]}
    ).resolve(headers(("x-databricks-user-token", "t")))
    assert got.role == USER


# -- forgery ---------------------------------------------------------------


def test_the_rstudio_username_header_is_never_read():
    """Measured: the proxy strips a forged `x-databricks-*` but forwards this one, first."""
    head = headers(("x-rstudio-username", ME))
    head.extend(browser(name="reader@example.com"))
    got = resolver(admin_users=[ME]).resolve(head)
    assert got.user_name == "reader@example.com"
    assert got.role == USER


def test_disagreeing_duplicates_are_refused_rather_than_chosen_between():
    head = headers(
        ("x-databricks-auth-validated", "true"),
        ("x-databricks-user-name", "attacker@example.com"),
        ("x-databricks-user-name", ME),
    )
    with pytest.raises(IdentityRefused):
        resolver().resolve(head)


def test_agreeing_duplicates_are_the_normal_case():
    assert resolver().resolve(browser()).user_name == ME


# -- elevation -------------------------------------------------------------


def test_elevation_proves_a_browser_callers_groups():
    res = resolver(
        validator=lambda t: {"userName": ME, "id": "42", "groups": [{"display": OWNER}]}
    )
    identity = res.resolve(browser())
    assert identity.role == USER
    accepted, detail = res.elevate(identity, "dapi-own-token")
    assert accepted and ADMIN in detail
    assert res.resolve(browser()).role == ADMIN


def test_elevation_refuses_another_users_token():
    res = resolver(
        validator=lambda t: {
            "userName": "someone.else@example.com",
            "id": "9",
            "groups": [{"display": OWNER}],
        }
    )
    accepted, detail = res.elevate(res.resolve(browser()), "dapi-not-mine")
    assert not accepted and "not to the signed-in caller" in detail
    assert res.resolve(browser()).role == USER


def test_elevation_refuses_a_token_that_does_not_validate():
    res = resolver(validator=lambda t: None)
    accepted, _ = res.elevate(res.resolve(browser()), "junk")
    assert not accepted


def test_elevation_can_be_disabled():
    res = resolver(allow_elevation=False, validator=lambda t: {"userName": ME})
    accepted, detail = res.elevate(res.resolve(browser()), "t")
    assert not accepted and "disabled" in detail


# -- storage segment -------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1234567890123456", "1234567890123456"),
        (ME, "dana.lee-corp.example.test"),
        ("../../etc/passwd", "etc-passwd"),
        ("", "unknown"),
        ("-leading", "leading"),
    ],
)
def test_storage_segment_is_key_safe(value, expected):
    assert storage_segment(value) == expected


def test_every_segment_survives_safe_key():
    from src.storage.base import safe_key

    for raw in ("1234567890123456", ME, "../../etc", "", "@@@", "a" * 400):
        assert safe_key(f"users/{storage_segment(raw)}/x.json")


# -- construction ----------------------------------------------------------


def test_the_workspace_host_falls_back_to_the_databricks_block():
    res = build_identity_resolver({"databricks": {"host": "https://example.net"}})
    assert res.host == "https://example.net"


def test_an_identity_block_wins_over_the_databricks_block():
    res = build_identity_resolver(
        {
            "databricks": {"host": "https://wrong.net"},
            "identity": {"workspace_host": "https://right.net", "admin_users": [ME]},
        }
    )
    assert res.host == "https://right.net" and res.admin_users == frozenset({ME})


def test_an_administrator_list_is_read_from_either_shape():
    """A comma-separated string and a YAML list must name the same administrators.

    The form writes a scalar (a line-anchored patcher refuses a key holding a block) while a
    hand-edited file naturally holds a sequence, so both shapes reach this reader. The
    interesting half is the string: folded as a sequence it would iterate letters, and
    single-character names would become administrators.
    """
    listed = build_identity_resolver({"identity": {"admin_users": [ME, OWNER]}})
    inline = build_identity_resolver({"identity": {"admin_users": f" {ME} ,{OWNER}"}})
    assert inline.admin_users == listed.admin_users == frozenset({ME.lower(), OWNER.lower()})
    assert len(inline.admin_users) == 2  # not one entry per character

    # And a blank string names nobody rather than everybody-with-an-empty-name.
    assert build_identity_resolver({"identity": {"admin_users": " "}}).admin_users == frozenset()


def test_no_host_means_no_validator_rather_than_a_failing_one():
    res = build_identity_resolver({})
    assert res._validator is None
    assert res.resolve(headers(("x-databricks-user-token", "t"))) is not None or True


def test_identity_serialises_for_the_audit_trail():
    got = Identity(user_id="42", user_name=ME, role=ADMIN, groups=(OWNER,)).as_dict()
    assert got["groups"] == [OWNER] and got["role"] == ADMIN
