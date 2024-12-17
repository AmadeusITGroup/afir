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

        **Analyze the provided incident details and generate a structured analysis with actionable insights:**
        
        **Incident Information:**  
        - **Incident ID:** {incident_id}  
        - **Timestamp:** {timestamp}  
        - **Description:** {description}  
        
        **Your analysis must include all the following sections:**  
        1. **Incident Summary:** A concise but thorough overview of the incident (be sure to include the timestamps).
        2. **Impact Assessment:** Evaluate the potential impact and assign a severity score on a scale of 1-10, with reasoning for the score.  
        3. **Key Investigation Areas:** Highlight the critical elements or anomalies that require immediate attention.  
        4. **Log Sources to Review:** Identify specific logs or systems (e.g., application logs, network traffic, user access logs) that may provide relevant insights.  
        5. **Initial Hypotheses:** Propose potential explanations for the incident, such as possible fraud patterns, system vulnerabilities, or suspicious activities.  
        6. **Recommended Actions:** Outline immediate steps to contain or mitigate the issue, including specific tasks or changes.  
        7. **Stakeholder Notification:** List the stakeholders (e.g., departments, individuals) who should be informed and explain their relevance to the incident.  
        
        **Contextual Considerations:**  
        - Use the provided details and relate them to any similar past incidents or known fraud patterns.  
        - Highlight any trends or correlations that could aid in understanding the incident further.  
        
        **Output Format:**  
        Provide your response as a JSON object.  
        
        **Additional Requirements:**  
        - Ensure all special characters, particularly those that may conflict with APIs (e.g., quotation marks, slashes, newlines), are properly escaped.  
        - Use concise and clear language to maximize readability and accuracy.  
        
        **Objective:** Provide a comprehensive and actionable analysis of the incident based on the given data and context.  
        
        """

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
