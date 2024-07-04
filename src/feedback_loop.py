import asyncio
import json
from src.utils.llm_utils import get_llm_response


class FeedbackLoop:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.feedback_data = []

    async def collect_feedback(self, incident_id, investigation_result, human_feedback):
        feedback_entry = {
            'incident_id': incident_id,
            'investigation_result': investigation_result,
            'human_feedback': human_feedback
        }
        self.feedback_data.append(feedback_entry)

        if len(self.feedback_data) >= 10:  # Process feedback in batches
            await self.process_feedback()

    async def process_feedback(self):
        prompt = f"""
        Analyze the following feedback data and provide insights on how to improve the fraud investigation system:

        {json.dumps(self.feedback_data, indent=2)}

        Please provide:
        1. Common patterns in successful detections of anomalies
        2. Types of anomalies that are frequently missed or misclassified
        3. Suggestions for improving the accuracy and relevance of detected anomalies
        4. Recommendations for adjusting confidence thresholds or scoring
        5. Insights on the effectiveness of recommended actions for each anomaly type
        6. Any new fraud patterns or techniques that should be incorporated into the anomaly detection process
        7. Suggestions for improving the overall investigation process based on human feedback

        Format your response as a JSON object with clear sections for each type of insight or recommendation.
        """

        insights = await get_llm_response(prompt, self.llm_config)
        await self.apply_insights(insights)
        self.feedback_data = []  # Clear processed feedback

    async def apply_insights(self, insights):
        # This method would implement the insights to improve the system
        # For example, updating LLM prompts, adjusting anomaly detection thresholds, etc.
        insights_dict = json.loads(insights)

        print("Applying insights to improve the system:")
        for category, recommendations in insights_dict.items():
            print(f"\n{category.capitalize()}:")
            for recommendation in recommendations:
                print(f"- {recommendation}")

        # TODO: Implement actual system improvements based on insights
        # For example:
        # - Update anomaly detection prompts based on common patterns
        # - Adjust confidence thresholds
        # - Update training data for fraud prediction model
        # - Modify report generation to highlight frequently missed anomalies


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

    feedback_loop = FeedbackLoop(llm_config)

    # Simulate collecting feedback
    await feedback_loop.collect_feedback(
        'INC-12345',
        {
            'anomalies': [
                {
                    'description': 'Multiple failed login attempts from various IP addresses',
                    'confidence_score': 0.95,
                    'implications': 'Possible brute force attack attempt',
                    'recommended_actions': 'Implement IP-based rate limiting and notify the account owner'
                }
            ]
        },
        {
            'accuracy': 4,
            'completeness': 5,
            'actionable_insights': 4,
            'missed_anomalies': ['Unusual transaction pattern after successful login'],
            'comments': 'Good detection of login attempts, but missed subsequent unusual activity'
        }
    )

    # Force processing of feedback
    await feedback_loop.process_feedback()

if __name__ == "__main__":
    asyncio.run(main())