import asyncio
import json
import logging

from utils.error_handling import async_retry_with_backoff
from utils.llm_utils import get_llm_response

logger = logging.getLogger(__name__)


def preprocess_logs(logs):
    combined_logs = []
    for source, entries in logs.items():
        for entry in entries:
            combined_logs.append(f"[{source}] {json.dumps(entry)}")

    max_chars = 15000  # Adjust based on LLM token limit
    combined_logs_str = "\n".join(combined_logs)
    if len(combined_logs_str) > max_chars:
        combined_logs_str = combined_logs_str[:max_chars] + "... [truncated]"

    return combined_logs_str


def parse_llm_response(llm_response):
    try:
        anomalies = json.loads(llm_response)
        if not isinstance(anomalies, list):
            raise ValueError("LLM response is not a JSON array")
        return anomalies
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {str(e)}")
        return []


class AnomalyDetectionModule:
    def __init__(self, config, llm_config, rag):
        self.config = config
        self.llm_config = llm_config
        self.rag = rag
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

        Use the provided context to enhance your analysis. Consider any similar past incidents, known fraud patterns, 
        or relevant information from the knowledge base. Be as detailed as possible.
        
        Be as cautious as possible.
        
        The fields of every anomaly must be description, supporting_data, potential_implications, confidence_score, 
        recommended_actions, patterns.

        Format your response as a JSON array of anomaly objects.
        Ensure that the generated JSONs are well-formed, properly escaped, and follow the specified 
        structure without any additional text output. Validate the JSONs structure before returning the result.
        """

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def detect(self, logs, understanding):
        try:
            combined_logs = preprocess_logs(logs)

            prompt = self.prompt_template.format(
                incident_understanding=json.dumps(understanding['analysis'], indent=2),
                log_data=combined_logs
            )

            if self.rag is None:
                prompt = self.llm_config['context'] + prompt

            llm_response = await get_llm_response(prompt, self.llm_config, self.rag)

            anomalies = parse_llm_response(llm_response)

            filtered_anomalies = self.filter_anomalies(anomalies)

            logger.info(f"Detected {len(filtered_anomalies)} anomalies for incident {understanding['incident_id']}")
            return filtered_anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies for incident {understanding['incident_id']}: {str(e)}")
            raise

    def filter_anomalies(self, anomalies):
        return [
            anomaly for anomaly in anomalies
            if anomaly.get('confidence_score', 0) >= self.config['threshold']
        ]


# Example usage
async def main():
    return


if __name__ == "__main__":
    asyncio.run(main())
