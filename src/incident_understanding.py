import asyncio
import json
import logging

from utils.error_handling import async_retry_with_backoff
from utils.llm_utils import get_llm_response

logger = logging.getLogger(__name__)


def structure_understanding(raw_understanding):
    try:
        return json.loads(raw_understanding)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON. Returning raw output.")
        return {"raw_output": raw_understanding}


class IncidentUnderstandingModule:
    def __init__(self, llm_config, rag):
        self.llm_config = llm_config
        self.rag = rag
        self.prompt_template = """

        Analyze the following incident and provide a detailed understanding:
        
        Incident ID: {incident_id}
        Timestamp: {timestamp}
        Description: {description}
        
        Please provide:
        1. A summary of the incident
        2. Potential impact and severity (on a scale of 1-10)
        3. Key elements to investigate
        4. Relevant log sources to check
        5. Any initial hypotheses about the nature of the incident (e.g., potential fraud patterns, suspicious activities)
        6. Recommended immediate actions
        7. Potential stakeholders to be notified
        
        Use the provided context to enhance your analysis. Consider any similar past incidents or known fraud patterns.
        
        Format your response as a JSON object with clearly labeled sections.
        Make sure to use escaping with all the symbols that may be a problem if I am using APIs.
        Don't output anything else except the JSON.\""""

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def process(self, incident):
        try:
            prompt = self.prompt_template.format(
                incident_id=incident['id'],
                timestamp=incident['timestamp'],
                description=incident['description']
            )

            if self.rag is None:
                prompt = self.llm_config['context'] + prompt

            understanding = await get_llm_response(prompt, self.llm_config, self.rag)
            structured_understanding = structure_understanding(understanding)

            logger.info(f"Processed understanding for incident {incident['id']}")
            return {
                "incident_id": incident['id'],
                "analysis": structured_understanding
            }
        except Exception as e:
            logger.error(f"Error processing incident {incident['id']}: {str(e)}")
            raise


# Example usage
async def main():
    return


if __name__ == "__main__":
    asyncio.run(main())
