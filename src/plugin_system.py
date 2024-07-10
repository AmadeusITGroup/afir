import importlib
import os
import logging
import yaml

logger = logging.getLogger(__name__)


class PluginManager:
    def __init__(self, plugin_dir):
        self.plugin_dir = plugin_dir
        self.plugins = {}
        self.active_plugins = []
        self.load_plugins()
        self.load_plugin_config()

    def load_plugins(self):
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f'{self.plugin_dir}.{module_name}')
                    if hasattr(module, 'register_plugin'):
                        plugin_info = module.register_plugin()
                        self.plugins[plugin_info['name']] = plugin_info
                        logger.info(f"Loaded plugin: {plugin_info['name']}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {module_name}: {str(e)}")

    def load_plugin_config(self):
        config_path = os.path.join('config', 'plugin_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.active_plugins = config.get('active_plugins', [])
        else:
            logger.warning("Plugin configuration file not found. All plugins will be active.")
            self.active_plugins = list(self.plugins.keys())

    def get_active_plugins(self):
        return [plugin for plugin in self.active_plugins if plugin in self.plugins]

    async def execute_plugin(self, plugin_name, *args, **kwargs):
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        plugin = self.plugins[plugin_name]
        try:
            return await plugin['execute'](*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing plugin '{plugin_name}': {str(e)}")
            raise


# Example plugin file (plugins/example_plugin.py)
"""
def analyze_data(incident, understanding, logs, anomalies):
    # Perform some analysis
    result = "Example plugin analysis result"
    return result

def register_plugin():
    return {
        'name': 'example_plugin',
        'description': 'An example plugin that performs additional analysis',
        'execute': analyze_data
    }
"""

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plugin_manager = PluginManager('plugins')
    print("Active plugins:", plugin_manager.get_active_plugins())

    # Simulate executing a plugin
    import asyncio


    async def test_plugin():
        try:
            result = await plugin_manager.execute_plugin('example_plugin',
                                                         incident={},
                                                         understanding={},
                                                         logs={},
                                                         anomalies=[])
            print("Plugin result:", result)
        except ValueError as e:
            print(str(e))


    asyncio.run(test_plugin())
