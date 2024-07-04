import unittest
from unittest.mock import patch
from src.api_call_generator import ApiCallGenerator


class TestApiCallGenerator(unittest.TestCase):
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
            }
        }
        self.api_call_generator = ApiCallGenerator(self.llm_config)

    @patch('utils.llm_utils.get_llm_response')
    async def test_generate(self, mock_get_llm_response):
        mock_get_llm_response.return_value = '''
        [
            {
                "target_log_source": "application_logs",
                "query_parameters": {
                    "start_time": "2023-07-05T00:00:00Z",
                    "end_time": "2023-07-05T23:59:59Z",
                    "search_terms": "login AND (failure OR suspicious)"
                },
                "context": "Investigate login failures"
            }
        ]
        '''
        understanding = {
            'incident_id': 'INC-12345',
            'analysis': 'Potential brute force attack detected'
        }
        result = await self.api_call_generator.generate(understanding)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['target_log_source'], 'application_logs')
        self.assertIn('start_time', result[0]['query_parameters'])
