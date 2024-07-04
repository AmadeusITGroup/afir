import asyncio
import json
from src.utils.llm_utils import get_llm_response
from src.utils.error_handling import async_retry_with_backoff
import logging

logger = logging.getLogger(__name__)


class AnomalyDetectionModule:
    def __init__(self, config, llm_config):
        self.config = config
        self.llm_config = llm_config
        self.prompt_template = """
        Analyze the following log data and incident understanding to detect any anomalies, suspicious patterns, or indicators of fraud:

        Incident Understanding:
        {incident_understanding}

        Log Data:
        {log_data}

        Please provide a comprehensive analysis considering various fraud scenarios and anomaly types. For each detected anomaly, provide:
        1. A detailed description of the anomaly
        2. The relevant log entries or data points supporting this anomaly
        3. The potential implications or risks associated with this anomaly
        4. A confidence score (0-1) indicating how certain you are that this is a genuine anomaly
        5. Recommended actions to investigate or mitigate this anomaly
        6. Any patterns or trends that might be relevant across multiple log entries

        Be sure to consider various types of anomalies, including but not limited to:
        - Unusual access patterns or login attempts
        - Suspicious transactions or financial activities
        - Abnormal system or user behaviors
        - Potential data breaches or information leaks
        - Unusual network traffic or communication patterns
        - Inconsistencies in user profiles or account activities

        Format your response as a JSON array of anomaly objects.
        """

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def detect(self, logs, understanding):
        try:
            combined_logs = self.preprocess_logs(logs)

            prompt = self.prompt_template.format(
                incident_understanding=json.dumps(understanding['analysis'], indent=2),
                log_data=combined_logs
            )

            llm_response = await get_llm_response(prompt, self.llm_config)

            anomalies = self.parse_llm_response(llm_response)

            filtered_anomalies = self.filter_anomalies(anomalies)

            logger.info(f"Detected {len(filtered_anomalies)} anomalies for incident {understanding['incident_id']}")
            return filtered_anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies for incident {understanding['incident_id']}: {str(e)}")
            raise

    def preprocess_logs(self, logs):
        combined_logs = []
        for source, entries in logs.items():
            for entry in entries:
                combined_logs.append(f"[{source}] {json.dumps(entry)}")

        max_chars = 15000  # Adjust based on LLM token limit
        combined_logs_str = "\n".join(combined_logs)
        if len(combined_logs_str) > max_chars:
            combined_logs_str = combined_logs_str[:max_chars] + "... [truncated]"

        return combined_logs_str

    def parse_llm_response(self, llm_response):
        try:
            anomalies = json.loads(llm_response)
            if not isinstance(anomalies, list):
                raise ValueError("LLM response is not a JSON array")
            return anomalies
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return []

    def filter_anomalies(self, anomalies):
        return [
            anomaly for anomaly in anomalies
            if anomaly.get('confidence_score', 0) >= self.config['threshold']
        ]


# Example usage
async def main():
    config = {
        'threshold': 0.6
    }

    llm_config = {
        'provider': 'openai',
        'models': {
            'default': {
                'name': 'gpt-4',
                'api_key': 'your_api_key_here',
                'max_tokens': 2000,
                'temperature': 0.7
            }
        }
    }

    logs = {
        'application_logs': [
            {'timestamp': '2023-07-07T12:00:00Z', 'message': 'User login failed', 'user_id': 'user123',
             'ip_address': '192.168.1.100'},
            {'timestamp': '2023-07-07T12:01:00Z', 'message': 'User login successful', 'user_id': 'user123',
             'ip_address': '10.0.0.1'},
            {'timestamp': '2023-07-07T12:02:00Z', 'message': 'Large transaction initiated', 'user_id': 'user123',
             'amount': 50000},
        ],
        'network_logs': [
            {'timestamp': '2023-07-07T12:00:05Z', 'source_ip': '192.168.1.100', 'destination_ip': '10.0.0.50',
             'port': 443},
            {'timestamp': '2023-07-07T12:01:30Z', 'source_ip': '10.0.0.1', 'destination_ip': '192.168.1.200',
             'port': 8080},
        ]
    }

    understanding = {
        'incident_id': 'INC-12345',
        'analysis': {
            'summary': 'Potential unauthorized access and suspicious transaction detected for user123.',
            'severity': 8,
            'key_elements': ['failed login', 'successful login from different IP', 'large transaction'],
            'relevant_log_sources': ['application_logs', 'network_logs']
        }
    }

    anomaly_detection = AnomalyDetectionModule(config, llm_config)
    anomalies = await anomaly_detection.detect(logs, understanding)

    print(json.dumps(anomalies, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
