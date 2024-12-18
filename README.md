<div align="center">

![Fraud Investigation System](assets/fraud_investigation_system.png)

</div>

The content of this repository comprises the work which is being been developed in the scope of RESCUE (RESilient Cloud for EUropE) project. 
The objective is to develop reusable, modular components to strengthen reliability and recover capabilities for (critical) digital services. Pilot Cyber Resilient Digital Twins for Data Centers and Edges that use open cloud infrastructure and are capable of hosting mission-critical applications at large scale.

# Automated Fraud Investigation and Reporting system

This project implements an advanced automated system for fraud investigation and reporting using Large Language Models (LLMs) and machine learning techniques. The system is designed to process incidents, analyze logs, detect anomalies, generate comprehensive reports, and continuously improve its performance.

## System Architecture

Below is a high-level overview of the Fraud Investigation System architecture:

```
+-------------------+      +------------------------+
|   Incident Input  |----->|  Incident Understanding|
+-------------------+      +------------------------+
                                       |
                                       v
+-------------------+      +------------------------+
|    Knowledge Base |<---->|     RAG Processing     |
+-------------------+      +------------------------+
                                       |
                                       v
+-------------------+      +------------------------+
|   Log Retrieval   |<---->|   Anomaly Detection    |
+-------------------+      +------------------------+
                                       |
                                       v
+-------------------+      +------------------------+
| Report Generation |<-----|    Output Interface    |
+-------------------+      +------------------------+
        |
        v
+-------------------+
|  Feedback Loop    |
+-------------------+
```

## Features

- Automated incident understanding using LLMs :white_check_mark:
- Dynamic API call generation for log retrieval :black_square_button:
- Asynchronous log retrieval from multiple sources (for now: Elasticsearch) :black_square_button:
- LLM-powered anomaly detection with statistical analysis :white_check_mark:
- Automated report generation :white_check_mark:
- Flexible output interface (email, file, API) :black_square_button:
- Rate limiting and input validation :white_check_mark:
- Caching and performance optimization :black_square_button:
- Robust error handling and retrying mechanisms :white_check_mark:
- Feedback loop for continuous improvement :black_square_button:
- Plugin system for easy extension of functionality :white_check_mark:
- Export of investigation results in various formats (JSON, CSV, XML, Excel) :white_check_mark:
- External API for submitting incidents :white_check_mark:

## System Architecture

The system consists of the following main components:

1. Incident Input Interface
2. Incident Understanding Module
3. API Call Generator
4. Log Retrieval Engine
5. Anomaly Detection Module
6. Report Generation Module
7. Output Interface
8. Plugin System
9. Feedback Loop
10. Performance Dashboard
11. Result Exporter
12. External API

These components are supported by utility modules for LLM integration, error handling, caching, performance optimization, input validation, and rate limiting.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/AmadeusITGroup/afir.git
   cd afir
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure the system by editing the YAML files in the `config/` directory (you can find some templates in `config/templates/`:
   - `main_config.yaml`: Main system configuration
   - `llm_config.yaml`: LLM provider configuration
   - `logging_config.yaml`: Logging configuration
   - `plugin_config.yaml`: Plugin configuration

## Usage

1. Start the main system:
   ```
   python src/main.py
   ```

2. Contact the system using API calls as descripted in `docs/user_guide.md`.

## Extending the System

### Adding Plugins

1. Create a new Python file in the `plugins/` directory.
2. Implement your plugin logic and a `register_plugin()` function.
3. Configure the plugin in the plugin configuration file.
4. The plugin will be automatically loaded by the PluginManager.

## Testing

Run the unit tests using:

```
python -m unittest discover tests
```

## Logging

The system uses Python's built-in logging module. Log files are stored in the `logs/` directory. You can adjust the logging level and output format in the `logging_config.yaml` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
