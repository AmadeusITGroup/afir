<div align="center">

![Fraud Investigation System](assets/fraud_investigation_system.png)

</div>

This has been developed in the scope of RESCUE (RESilient Cloud for EUropE) project. 
The objective is to develop reusable, modular components to strengthen reliability and recover capabilities for (critical) digital services. Pilot Cyber Resilient Digital Twins for Data Centers and Edges that use open cloud infrastructure and are capable of hosting mission-critical applications at large scale.

# Automated Fraud Investigation and Reporting system

This project implements an advanced automated system for fraud investigation and reporting using Large Language Models (LLMs) and machine learning techniques. The system is designed to process incidents, analyze logs, detect anomalies, generate comprehensive reports, and continuously improve its performance.

## Features

- Automated incident understanding using LLMs
- Dynamic API call generation for log retrieval
- Asynchronous log retrieval from multiple sources (Elasticsearch, Splunk)
- LLM-powered anomaly detection with statistical analysis
- Automated report generation with visualizations
- Flexible output interface (email, file, API)
- Rate limiting and input validation
- Caching and performance optimization
- Robust error handling and retrying mechanisms
- Web interface for monitoring and manual intervention
- Feedback loop for continuous improvement
- Plugin system for easy extension of functionality
- Performance dashboard for visualizing system performance and trends
- Export of investigation results in various formats (JSON, CSV, XML, Excel)
- External API for submitting incidents and retrieving results

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

4. Set up environment variables for API keys and sensitive information:
   ```
   export OPENAI_API_KEY=your_openai_api_key
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   export HF_API_KEY=your_huggingface_api_key
   export ES_USERNAME=your_elasticsearch_username
   export ES_PASSWORD=your_elasticsearch_password
   export SPLUNK_TOKEN=your_splunk_token
   export SMTP_USERNAME=your_smtp_username
   export SMTP_PASSWORD=your_smtp_password
   ```

5. Configure the system by editing the YAML files in the `config/` directory:
   - `main_config.yaml`: Main system configuration
   - `llm_config.yaml`: LLM provider configuration

6. Set up a Redis server for caching (optional):
   ```
   sudo apt-get install redis-server
   sudo systemctl start redis-server
   ```

## Usage

1. Start the main system:
   ```
   python src/main.py
   ```

2. Run the web interface:
   ```
   python src/web_interface.py
   ```

3. Start the performance dashboard:
   ```
   python src/dashboard.py
   ```

4. Run the external API:
   ```
   python src/api.py
   ```

## Extending the System

### Adding Plugins

1. Create a new Python file in the `plugins/` directory.
2. Implement your plugin logic and a `register_plugin()` function.
3. The plugin will be automatically loaded by the PluginManager.

### Customizing the Feedback Loop

Modify the `process_feedback()` method in `src/feedback_loop.py` to implement custom logic for applying insights and improving the system.

### Adding New Export Formats

Extend the `ResultExporter` class in `src/export_results.py` to add new export formats.

## API Documentation

### Submit Incident

- Endpoint: `/api/submit_incident`
- Method: POST
- Payload: JSON object with incident details
- Response: Confirmation message with incident ID

### Get Result

- Endpoint: `/api/get_result/<incident_id>`
- Method: GET
- Response: Investigation result for the specified incident

### System Status

- Endpoint: `/api/system_status`
- Method: GET
- Response: Current status of the fraud investigation system

## Testing

Run the unit tests using:

```
python -m unittest discover tests
```

## Logging

The system uses Python's built-in logging module. Log files are stored in the `logs/` directory. You can adjust the logging level and output format in the `logging_config.yaml` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
