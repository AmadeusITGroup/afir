import asyncio
from src.utils.llm_utils import get_llm_response
from src.utils.error_handling import async_retry_with_backoff
import logging
import json

logger = logging.getLogger(__name__)

class ApiCallGenerator:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.prompt_template = """
        Based on the following incident understanding, generate a list of API calls to retrieve relevant logs:
        
        {understanding}
        
        For each API call, provide:
        1. The target log source (e.g., "application_logs", "security_logs")
        2. The query parameters (e.g., time range, filters, search terms)
        3. Any additional context or justification for this API call
        
        Consider various log sources that might be relevant, such as:
        - Application logs
        - Security logs
        - Network traffic logs
        - Database transaction logs
        - User activity logs
        
        Format the output as a list of JSON objects, each representing an API call.
        """

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def generate(self, understanding):
        try:
            prompt = self.prompt_template.format(understanding=json.dumps(understanding['analysis'], indent=2))

            api_calls_str = await get_llm_response(prompt, self.llm_config)
            api_calls = self.parse_api_calls(api_calls_str)

            logger.info(f"Generated {len(api_calls)} API calls for incident {understanding['incident_id']}")
            return api_calls
        except Exception as e:
            logger.error(f"Error generating API calls for incident {understanding['incident_id']}: {str(e)}")
            raise

    def parse_api_calls(self, api_calls_str):
        try:
            api_calls = json.loads(api_calls_str)
            if not isinstance(api_calls, list):
                raise ValueError("API calls must be a list")
            return [self.validate_api_call(call) for call in api_calls]
        except json.JSONDecodeError:
            logger.error("Failed to parse API calls JSON")
            return []

    def validate_api_call(self, api_call):
        required_fields = ['target_log_source', 'query_parameters', 'context']
        for field in required_fields:
            if field not in api_call:
                logger.warning(f"API call missing required field: {field}")
                api_call[field] = "Not provided"
        return api_call

# Example usage
async def main():
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

    understanding = {
        'incident_id': 'INC-12345',
        'analysis': {
            'summary': 'Multiple failed login attempts followed by a large transaction from a new IP address.',
            'severity': 8,
            'key_elements': ['login attempts', 'large transaction', 'new IP address'],
            'relevant_log_sources': ['authentication logs', 'transaction logs', 'network logs']
        }
    }

    api_call_generator = ApiCallGenerator(llm_config)
    api_calls = await api_call_generator.generate(understanding)
    print(json.dumps(api_calls, indent=2))

if __name__ == "__main__":
    asyncio.run(main())