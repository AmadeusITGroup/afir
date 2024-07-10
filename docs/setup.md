# Fraud Investigation System Setup Guide

## Prerequisites
- Python 3.8+
- Redis (for caching and message queuing)
- Elasticsearch (for log storage and retrieval)
- Splunk (for security log analysis)

## Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/your-org/fraud-investigation-system.git
   cd fraud-investigation-system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the configuration files in the `config/` directory. Refer to `configuration.md` for details.

5. Start the main application:
   ```
   python src/main.py
   ```

## Running Tests

To run the test suite:
```
python -m unittest discover tests
```

## Troubleshooting

If you encounter any issues during setup, please refer to our FAQ section or open an issue on the GitHub repository.