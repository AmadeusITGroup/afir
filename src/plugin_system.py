import importlib
import importlib.util
import logging
import os

import yaml

from utils.paths import config_path, plugins_dir

logger = logging.getLogger(__name__)


class PluginManager:
    def __init__(self, plugin_dir=None):
        # The argument is accepted for compatibility and ignored: the directory always
        # anchors to the repo's plugins/ so loading works from any cwd.
        self.plugin_dir = str(plugins_dir())
        self.plugins = {}
        self.active_plugins = []
        self.load_plugin_config()
        self.load_plugins()

    def load_plugins(self):
        if not os.path.isdir(self.plugin_dir):
            logger.warning("Plugin directory not found: %s", self.plugin_dir)
            return
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                if module_name not in self.active_plugins:
                    continue
                try:
                    module = self._import_plugin(module_name, filename)
                    if hasattr(module, "register_plugin"):
                        plugin_info = module.register_plugin()
                        self.plugins[plugin_info["name"]] = plugin_info
                        logger.info(f"Loaded plugin: {plugin_info['name']}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {module_name}: {str(e)}")

    def _import_plugin(self, module_name, filename):
        """Import a plugin by file path so it works regardless of sys.path / cwd."""
        path = os.path.join(self.plugin_dir, filename)
        spec = importlib.util.spec_from_file_location(
            f"afir_plugins.{module_name}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_plugin_config(self):
        config_file = config_path("plugin_config.yaml")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f) or {}
                # `or []` rather than a `.get` default: the shipped template lists every
                # plugin commented out, so `active_plugins:` is a present key with a null
                # value and the default never applies.
                self.active_plugins = config.get("active_plugins") or []
        else:
            logger.warning(
                "Plugin configuration file not found. No plugins will be active."
            )
            self.active_plugins = []

    def get_active_plugins(self):
        return [plugin for plugin in self.active_plugins if plugin in self.plugins]

    async def execute_plugin(self, plugin_name, *args, **kwargs):
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        plugin = self.plugins[plugin_name]
        try:
            return await plugin["execute"](*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing plugin '{plugin_name}': {str(e)}")
            raise
