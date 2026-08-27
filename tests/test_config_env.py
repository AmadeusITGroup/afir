"""Tests for ${VAR} / ${VAR:-default} config expansion in src.main._expand_env.

THE DEFAULT FORM IS A DEPLOYMENT MECHANISM, not a convenience. Where an App's durable
state lives cannot be configured from the UI — in a container the first config comes from
the read-only bundle, so a UI switch to the Volume backend is written to the disk that the
applying restart discards. `${VAR:-default}` is how `app.yaml` decides it on the FIRST
boot while a laptop that sets nothing keeps the shipped value, so the cases below are the
ones that would silently put a deployment back on ephemeral disk.
"""

import pytest

from src.main import _expand_env


def test_expand_env_resolves_set_var(monkeypatch):
    monkeypatch.setenv("ELK_USER", "alice")
    assert _expand_env("${ELK_USER}") == "alice"


def test_expand_env_unset_var_becomes_empty(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    assert _expand_env("${DEFINITELY_UNSET_VAR}") == ""


def test_expand_env_recurses_dict_and_list(monkeypatch):
    monkeypatch.setenv("ELK_USER", "alice")
    monkeypatch.setenv("ELK_PASS", "s3cret")
    cfg = {
        "backends": {
            "elasticsearch": {
                "secondary-prd": {
                    "url": "https://gw.example",
                    "username": "${ELK_USER}",
                    "password": "${ELK_PASS}",
                    "indices": ["a", "${ELK_USER}", "c"],
                }
            }
        }
    }
    out = _expand_env(cfg)
    es = out["backends"]["elasticsearch"]["secondary-prd"]
    assert es["username"] == "alice"
    assert es["password"] == "s3cret"
    assert es["indices"] == ["a", "alice", "c"]
    assert es["url"] == "https://gw.example"


def test_expand_env_leaves_plain_strings_and_env_name_keys(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "should-not-be-used")
    cfg = {
        "api_key_env": "DATABRICKS_TOKEN",  # env-var *name*, not a ${} ref
        "url": "https://adb.example.net",
        "warehouse_id": "abc123",
    }
    out = _expand_env(cfg)
    # env-var-name key preserved verbatim; only ${...} references are substituted.
    assert out["api_key_env"] == "DATABRICKS_TOKEN"
    assert out["url"] == "https://adb.example.net"
    assert out["warehouse_id"] == "abc123"


def test_expand_env_passes_non_strings_through():
    assert _expand_env(30) == 30
    assert _expand_env(True) is True
    assert _expand_env(None) is None


def test_expand_env_embedded_var(monkeypatch):
    monkeypatch.setenv("HOST", "gw.example")
    assert _expand_env("https://${HOST}/path") == "https://gw.example/path"


# --- ${VAR:-default} -------------------------------------------------------------


def test_a_set_var_wins_over_its_default(monkeypatch):
    monkeypatch.setenv("AFIR_STORAGE_BACKEND", "databricks")
    assert _expand_env("${AFIR_STORAGE_BACKEND:-local}") == "databricks"


def test_an_unset_var_yields_the_default(monkeypatch):
    """Which is what keeps every laptop and VM on the shipped value."""
    monkeypatch.delenv("AFIR_STORAGE_BACKEND", raising=False)
    assert _expand_env("${AFIR_STORAGE_BACKEND:-local}") == "local"


def test_a_declared_but_empty_var_also_yields_the_default(monkeypatch):
    """The load-bearing half, and the one a naive `os.environ.get(name, default)` fails.

    A platform can inject a declared-but-blank env var — an `app.yaml` entry left with no
    value, an app resource not filled in. Treating that as a value turns the default into
    "", and for `storage.backend` the empty string means LOCAL DISK in a container: every
    pending approval lost on the next restart, while the app reports itself healthy. The
    empty string is not something anyone configures on purpose, so it counts as unset —
    which is also what the shell does.
    """
    monkeypatch.setenv("AFIR_STORAGE_BACKEND", "")
    assert _expand_env("${AFIR_STORAGE_BACKEND:-local}") == "local"


def test_the_bare_form_keeps_its_credential_semantics(monkeypatch):
    """A missing SECRET must stay empty, not acquire a default.

    An unset `${ES_PASSWORD}` expands to "" so the source is skipped with a log line; a
    default would hand the retriever a literal that only 401s at query time. So the two
    forms must not converge.
    """
    monkeypatch.delenv("ES_PASSWORD", raising=False)
    assert _expand_env("${ES_PASSWORD}") == ""


def test_an_explicitly_empty_default_is_honoured(monkeypatch):
    """`${VAR:-}` is how a required-but-unknown value ships blank.

    `storage.databricks.catalog` uses it: blank makes the backend refuse to write and say
    so at boot, which is the intended unconfigured state.
    """
    monkeypatch.delenv("AFIR_UC_CATALOG", raising=False)
    assert _expand_env("${AFIR_UC_CATALOG:-}") == ""


@pytest.mark.parametrize(
    "template, expected",
    [
        ("/Volumes/${AFIR_UC_CATALOG:-c}/afir", "/Volumes/cat/afir"),
        ("${AFIR_UC_CATALOG:-c}/${MISSING_VAR:-fallback}", "cat/fallback"),
        # A default containing a colon or a dash must survive: only the FIRST `:-`
        # separates, and `}` ends the reference.
        ("${MISSING_VAR:-https://h:443/x}", "https://h:443/x"),
        ("${MISSING_VAR:-a-b-c}", "a-b-c"),
    ],
)
def test_defaults_compose_and_survive_punctuation(template, expected, monkeypatch):
    monkeypatch.setenv("AFIR_UC_CATALOG", "cat")
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert _expand_env(template) == expected


def test_the_shipped_templates_boot_a_local_deployment_unchanged(monkeypatch):
    """Every `${VAR:-default}` in the real templates, with NOTHING set.

    The templates are what an unconfigured checkout and a deployed bundle both start from,
    so a default that expanded to "" here would be an empty `pack_dir` or an empty
    `backend` on every laptop. Reads the shipped files rather than a fixture: the point is
    that *those* files are safe.
    """
    import re

    import yaml

    from src.utils.paths import REPO_ROOT

    for name in ("main_config.yaml", "llm_config.yaml"):
        text = (REPO_ROOT / "config" / "templates" / name).read_text(encoding="utf-8")
        for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}", text):
            monkeypatch.delenv(var, raising=False)
        expanded = _expand_env(yaml.safe_load(text))

        if name == "main_config.yaml":
            assert expanded["storage"]["backend"] == "local"
            assert expanded["storage"]["databricks"]["schema"] == "afir"
            assert expanded["storage"]["databricks"]["volume"] == "state"
            # Blank on purpose: the backend refuses to write and says so at boot.
            assert expanded["storage"]["databricks"]["catalog"] == ""
            assert expanded["knowledge"]["pack_dir"] == "mock_domain"
        else:
            # Blank = "Model Serving on the workspace the SDK resolves". A literal
            # placeholder here reached the SDK URL-encoded and 401'd every stage.
            assert expanded["base_url"] == ""
            assert expanded["model"]


def test_the_templates_env_drive_the_deployment_decisions(monkeypatch):
    """The other direction: what `app.yaml` sets must actually land.

    These four are the bootstrap set — the decisions an operator cannot make from the UI
    because they govern where the UI's own edits are stored.
    """
    import yaml

    from src.utils.paths import REPO_ROOT

    monkeypatch.setenv("AFIR_STORAGE_BACKEND", "databricks")
    monkeypatch.setenv("AFIR_UC_CATALOG", "some_catalog")
    monkeypatch.setenv("AFIR_UC_VOLUME", "afir_state")
    monkeypatch.setenv("AFIR_PACK_DIR", "some_pack")

    text = (REPO_ROOT / "config" / "templates" / "main_config.yaml").read_text(
        encoding="utf-8"
    )
    expanded = _expand_env(yaml.safe_load(text))

    assert expanded["storage"]["backend"] == "databricks"
    assert expanded["storage"]["databricks"]["catalog"] == "some_catalog"
    assert expanded["storage"]["databricks"]["volume"] == "afir_state"
    assert expanded["knowledge"]["pack_dir"] == "some_pack"
