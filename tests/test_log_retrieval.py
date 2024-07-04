import unittest
from unittest.mock import patch, MagicMock
from src.log_retrieval import LogRetrievalEngine


class TestLogRetrieval(unittest.TestCase):
    def setUp(self):
        self.log_sources = [
            {
                "name": "application_logs",
                "type": "elasticsearch",
                "url": "http://elasticsearch:9200",
                "username": "test_user",
                "password": "test_password",
                "timeout": 30
            },
            {
                "name": "security_logs",
                "type": "splunk",
                "url": "https://splunk.example.com:8089",
                "token": "test_token",
                "timeout": 30
            }
        ]
        self.log_retrieval = LogRetrievalEngine(self.log_sources)

    @patch('elasticsearch.AsyncElasticsearch')
    async def test_get_elasticsearch_logs(self, mock_elasticsearch):
        mock_es_instance = MagicMock()
        mock_elasticsearch.return_value = mock_es_instance
        mock_es_instance.search.return_value = {
            'hits': {
                'hits': [
                    {'_source': {'message': 'Test log 1'}},
                    {'_source': {'message': 'Test log 2'}}
                ]
            }
        }

        api_call = {
            'target_log_source': 'application_logs',
            'query_parameters': {
                'start_time': '2023-07-05T00:00:00Z',
                'end_time': '2023-07-05T23:59:59Z',
                'search_terms': 'test'
            }
        }

        result = await self.log_retrieval.get_elasticsearch_logs(api_call)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['message'], 'Test log 1')
