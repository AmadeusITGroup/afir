"""Tests for repo-anchored path resolution (``src/utils/paths.py``).

``resolve_model_path`` exists because a *relative* sentence-transformers path in
config only resolves when cwd happens to be the repo root. Launched from ``src/``
or ``scripts/`` it silently missed, and RAG degraded to the keyword
``PlaybookFallback`` with no error — the failure mode this guards against.
"""

from pathlib import Path

from src.utils.paths import REPO_ROOT, resolve_model_path


def test_bare_hf_model_id_unchanged():
    """A bare HF id must pass through so the library resolves it from the hub/cache."""
    assert resolve_model_path("all-mpnet-base-v2") == "all-mpnet-base-v2"


def test_relative_path_anchored_to_repo_root():
    """A path-like reference is anchored, so cwd no longer decides whether it resolves."""
    resolved = resolve_model_path("model_cache/all-mpnet-base-v2")
    assert Path(resolved).is_absolute()
    assert resolved == str(REPO_ROOT / "model_cache" / "all-mpnet-base-v2")


def test_absolute_path_unchanged():
    absolute = str(REPO_ROOT / "model_cache" / "all-mpnet-base-v2")
    assert resolve_model_path(absolute) == absolute


def test_resolution_is_cwd_independent(tmp_path, monkeypatch):
    """The regression: same input must resolve identically from any cwd."""
    from_root = resolve_model_path("model_cache/all-mpnet-base-v2")
    monkeypatch.chdir(tmp_path)
    assert resolve_model_path("model_cache/all-mpnet-base-v2") == from_root


# --- platform detection (``src/utils/deployment.py``) ------------------------
#
# Lives here rather than in its own file: it is the same class of fact as a resolved path
# — one environment reading that decides a default — and the failure mode is the same
# shape, a wrong answer that degrades behaviour without erroring.


def test_app_detection_reads_the_one_variable_the_platform_injects(monkeypatch):
    """``DATABRICKS_APP_PORT`` and nothing softer.

    Not a hostname and not the OAuth variables: a laptop with the SDK configured has
    those, and a local run that believes it is an App loses functionality no platform
    limit applies to. Blank counts as absent, because an env var set to "" is how a
    platform says nothing rather than how it says zero.
    """
    from src.utils.deployment import running_as_databricks_app

    monkeypatch.delenv("DATABRICKS_APP_PORT", raising=False)
    assert running_as_databricks_app() is False
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "abc")
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.azuredatabricks.net")
    assert running_as_databricks_app() is False, "OAuth vars are not an App"
    monkeypatch.setenv("DATABRICKS_APP_PORT", "   ")
    assert running_as_databricks_app() is False, "blank is absent"
    monkeypatch.setenv("DATABRICKS_APP_PORT", "8000")
    assert running_as_databricks_app() is True


def test_a_tri_state_defaults_to_the_platform_and_yields_to_the_operator():
    from src.utils.deployment import resolve_mode

    for absent in (None, "", "auto", "AUTO", "  auto  "):
        assert resolve_mode(absent, platform_default=True) is True
        assert resolve_mode(absent, platform_default=False) is False
    for on in ("on", "true", "yes", "enabled", "1", "TRUE"):
        assert resolve_mode(on, platform_default=False) is True
    for off in ("off", "false", "no", "disabled", "0"):
        assert resolve_mode(off, platform_default=True) is False


def test_an_unparseable_value_resolves_to_the_default_and_is_reportable():
    """Two functions rather than a raise, and the split is the point.

    These switches are read at boot on a path with no operator watching, so a typo must
    not be the reason the server does not come up — but resolving it silently is how a
    deliberate override becomes a mystery. So it resolves to the platform's answer, and
    the caller can tell that it did.
    """
    from src.utils.deployment import is_recognised_mode, resolve_mode

    assert resolve_mode("enabld", platform_default=True) is True
    assert resolve_mode("enabld", platform_default=False) is False
    assert is_recognised_mode("enabld") is False
    assert is_recognised_mode("auto") is True
    assert is_recognised_mode(None) is True
    assert is_recognised_mode("off") is True
