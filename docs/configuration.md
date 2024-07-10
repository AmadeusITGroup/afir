# Configuration Guide

The Fraud Investigation System uses YAML files for configuration. The main configuration files are located in the `config/` directory.

## main_config.yaml

This file contains the main system configuration:

```yaml
incident_input:
  endpoint: "/api/v1/incidents"
  port: 5000
  rate_limit:
    requests: 1000
    per_seconds: 3600

log_sources:
  - name: "application_logs"
    type: "elasticsearch"
    url: "http://elasticsearch:9200"
    index: "app_logs_*"
    username: "${ES_USERNAME}"
    password: "${ES_PASSWORD}"
  - name: "security_logs"
    type: "splunk"
    url: "https://splunk.example.com:8089"
    token: "${SPLUNK_TOKEN}"

anomaly_detection:
  threshold: 0.7

# ... other configurations
```

## llm_config.yaml

This file configures the LLM providers:

```yaml
provider: "openai"
models:
  default:
    name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    max_tokens: 2000
    temperature: 0.7
  fine_tuned:
    name: "ft:gpt-3.5-turbo-0613:acme-corp::7qCrVKmR"
    api_key: "${OPENAI_API_KEY}"
    max_tokens: 1000
    temperature: 0.5

# ... other LLM configurations
```

## plugin_config.yaml

This file configures the plugin system:

```yaml
active_plugins:
  - example_plugin

plugin_settings:
  example_plugin:
    setting1: value1
    setting2: value2
```

Replace placeholder values with your actual configuration details and API keys.