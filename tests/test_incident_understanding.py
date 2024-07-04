import unittest
from unittest.mock import patch
from src.incident_understanding import IncidentUnderstandingModule

class TestIncidentUnderstanding(unittest.TestCase):
    def setUp(self):
        self.llm_config = {
            'provider': 'openai',
            'models': {
                'default': {
                    'name': 'gpt-4',
                    'api_key': 'test_key',
                    'max_tokens': 2000,
                    'temperature': 0.7
                }
            },
            'prompt_templates': {
                'incident_understanding': 'Analyze this incident: {description}'
            }
        }
        self.incident_understanding = IncidentUnderstandingModule(self.llm_config)

    @patch('utils.llm_utils.get_llm_response')
    async def test_process(self, mock_get_llm_response):
        mock_get_llm_response.return_value = "This is a test analysis"
        incident = {
            'id': 'INC-12345',
            'timestamp': '2023-07-05T10:30:00',
            'description': 'Test incident'
        }
        result = await self.incident_understanding.process(incident)
        self.assertEqual(result['incident_id'], 'INC-12345')
        self.assertEqual(result['analysis'], "This is a test analysis")