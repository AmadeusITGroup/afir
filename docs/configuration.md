# Configuration Guide

The Fraud Investigation System uses YAML files for configuration. The main configuration files are located in the `config/` directory.
Some examples are available in `config/templates`.

## main_config.yaml

This file contains the main system configuration:

```yaml
anomaly_detection:
  use_llm: true
  threshold: 0.8
  max_anomalies: 50
  time_window:
    hours: 24

report_generation:
  template: "standard_report"
  output_path: '../exports'
  output_format: 'pdf'
  logo_path: '../assets/company_logo.jfif'
  max_report_size_mb: 10

output_interface:
  type: "email"
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  smtp_username: "${SMTP_USERNAME}"
  smtp_password: "${SMTP_PASSWORD}"
  sender_email: "fraud_detection@example.com"
  recipients:
    - "security_team@example.com"
    - "fraud_analysts@example.com"
  cc:
    - "management@example.com"
  attachment_size_limit_mb: 25

plugin_dir: "../plugins"

knowledge_base:
  path: "/path/to/your/knowledge_base"
  update_frequency: "daily"

# ... other configurations
```

## llm_config.yaml

This file configures the LLM providers:

```yaml
'provider': 'generic'
'use_fine_tuned': false
'url': "http://model.url"
'token': "token"
'models':
  'default':
    'name': "gpt-3.5-turbo-0613"
    'max_tokens': 2000
    'temperature': 0.7
'context': [YOUR CONTEXT IF NOT USING RAG]

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

## logging.config

This file configures the logging system:

```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: default
    filename: logs/fraud_investigation.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

# ... other configurations
```

Replace placeholder values with your actual configuration details and API keys.