import asyncio
import json
import logging

from src.utils.error_handling import async_retry_with_backoff
from src.utils.llm_utils import get_llm_response

logger = logging.getLogger(__name__)


class ApiCallGenerator:
    def __init__(self, config, llm_config):
        self.config = config
        self.llm_config = llm_config
        self.prompt_template = """Based on the following incident understanding, generate a list of API calls 
        containing only a single API call to retrieve relevant logs:
        
        {understanding}
        
        For the API call, provide only:
        1. A field target_log_source (choose the type of the logs from this list: {logs_names_list})
        2. A field officeId, the office ID
        3. A field userId, the ID of the suspected user ("*" if not available)
        4. A field date_from, the start date from when to retrieve the logs (in the format yyyy-MM-dd) ("*" if not available)
        5. A field date_to, the end date up to when to retrieve the logs (in the format yyyy-MM-dd) ("*" if not available)

        Format the output as a list of JSON objects, each representing an API call (even if it will contain just one 
        API call). Ensure that the generated JSONs are well-formed, properly escaped, and follow the specified 
        structure without any additional text output. Validate the JSON structure before returning the result."""

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def generate(self, understanding):
        try:
            log_types = ', '.join(self.config['names_list'])

            prompt = self.prompt_template.format(
                understanding=json.dumps(understanding['analysis'], indent=2),
                logs_names_list=log_types
            )

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
    return


if __name__ == "__main__":
    asyncio.run(main())
