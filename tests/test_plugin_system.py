"""`src/plugin_system.py` — construction against the config files that really ship.

WHAT THIS FILE IS GUARDING
==========================
`PluginManager` is constructed in `main()`'s `modules` dict and `test_main.py` supplies a
`MagicMock()` in its place, so until this file existed **nothing ever ran `__init__`**.
That is how a boot-killing `TypeError` shipped: `plugin_config.yaml` ships with every
plugin commented out, which makes `active_plugins:` a *present* key with a **null** value.
`config.get("active_plugins", [])` returns the default only for an absent key, so `None`
reached `module_name not in self.active_plugins` and the process died — on a deployed
bundle, where the template is the only config there is, before the server ever bound.

So the shape under test is a comment-only value, not a missing key. Both are asserted,
because they are two different YAML facts that a `.get` default conflates.
"""

import pytest
import yaml

from src import plugin_system
from src.plugin_system import PluginManager

#: The shipped template's shape, verbatim in the part that matters: the key is present and
#: its only content is a comment, so `safe_load` yields `{"active_plugins": None}`.
COMMENTED_OUT_TEMPLATE = """\
active_plugins:
  # - custom_anomaly_detection

plugin_settings:
  custom_anomaly_detection:
    threshold: 0.8
"""


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Redirect `config_path` so a test never reads the operator's real config."""
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setattr(plugin_system, "config_path", lambda name: root / name)
    monkeypatch.setattr(plugin_system, "plugins_dir", lambda: tmp_path / "plugins")
    (tmp_path / "plugins").mkdir()
    return root


def test_a_commented_out_active_plugins_list_is_not_none(config_dir):
    """`active_plugins:` with nothing but a comment under it parses to `None`.

    The template ships exactly this, so it is the first App boot's config. A `.get`
    default cannot help: the key is present.
    """
    (config_dir / "plugin_config.yaml").write_text(
        COMMENTED_OUT_TEMPLATE, encoding="utf-8"
    )
    # The premise, pinned: if PyYAML ever returned `[]` here the guard would be dead code
    # and this test would be passing for the wrong reason.
    parsed = yaml.safe_load(COMMENTED_OUT_TEMPLATE)
    assert parsed["active_plugins"] is None

    manager = PluginManager()

    assert manager.active_plugins == []
    assert manager.get_active_plugins() == []


def test_an_absent_key_and_an_absent_file_both_yield_no_plugins(config_dir):
    """The two cases the null one was mistaken for."""
    (config_dir / "plugin_config.yaml").write_text(
        "plugin_settings: {}\n", encoding="utf-8"
    )
    assert PluginManager().active_plugins == []

    (config_dir / "plugin_config.yaml").unlink()
    assert PluginManager().active_plugins == []


def test_a_named_plugin_is_loaded_and_an_unlisted_one_is_not(config_dir, tmp_path):
    """The positive path, so the guard above cannot be satisfied by loading nothing ever."""
    for name in ("wanted", "ignored"):
        (tmp_path / "plugins" / f"{name}.py").write_text(
            "async def _run(*a, **k):\n"
            "    return {}\n"
            "def register_plugin():\n"
            f"    return {{'name': '{name}', 'execute': _run}}\n",
            encoding="utf-8",
        )
    (config_dir / "plugin_config.yaml").write_text(
        "active_plugins:\n  - wanted\n", encoding="utf-8"
    )

    manager = PluginManager()

    assert manager.get_active_plugins() == ["wanted"]
    assert "ignored" not in manager.plugins
