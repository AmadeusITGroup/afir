import unittest
from unittest.mock import patch, MagicMock
from src.incident_input import IncidentInputInterface

class TestIncidentInput(unittest.TestCase):
    def setUp(self):
        self.config = {
            'endpoint': '/api/v1/incidents',
            'port': 5000,
            'rate_limit': {'requests': 10, 'per_seconds': 60}
        }
        self.incident_input = IncidentInputInterface(self.config)

    @patch('aiohttp.web.Request')
    async def test_receive_incident(self, mock_request):
        mock_request.json.return_value = {
            'id': 'INC-12345',
            'timestamp': '2023-07-05T10:30:00',
            'description': 'Test incident',
            'source': 'Test',
            'severity': 5
        }
        response = await self.incident_input.receive_incident(mock_request)
        self.assertEqual(response.status, 201)
        self.assertIn('Received incident: INC-12345', response.text)

    async def test_get_incidents(self):
        self.incident_input.incidents.put_nowait({'id': 'INC-12345'})
        self.incident_input.incidents.put_nowait({'id': 'INC-67890'})
        incidents = await self.incident_input.get_incidents(batch_size=2)
        self.assertEqual(len(incidents), 2)
        self.assertEqual(incidents[0]['id'], 'INC-12345')
        self.assertEqual(incidents[1]['id'], 'INC-67890')