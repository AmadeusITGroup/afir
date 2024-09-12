from aiohttp import web
import asyncio
from asyncio import Queue
import logging
from src.utils.performance import AsyncRateLimiter
from src.utils.error_handling import async_retry_with_backoff
import json

logger = logging.getLogger(__name__)


class IncidentInputInterface:
    def __init__(self, config):
        self.config = config
        self.app = web.Application()
        self.app.router.add_post(config['endpoint'], self.receive_incident)
        self.incidents = Queue()
        self.rate_limiter = AsyncRateLimiter(config['rate_limit']['requests'], config['rate_limit']['per_seconds'])

    async def start_server(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.config['port'])
        await site.start()
        logger.info(f"Incident input server started on port {self.config['port']}")

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def receive_incident(self, request):
        await self.rate_limiter.acquire()
        try:
            incident = await request.json()
            if not self.validate_incident(incident):
                return web.Response(status=400, text="Invalid incident data")

            await self.incidents.put(incident)
            logger.info(f"Received incident: {incident['id']}")
            return web.Response(status=201, text=f"Received incident: {incident['id']}")
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error(f"Error receiving incident: {str(e)}")
            return web.Response(status=500, text="Internal server error")

    def validate_incident(self, incident):
        required_fields = ['id', 'timestamp', 'description']
        return all(field in incident for field in required_fields)

    async def get_incidents(self, batch_size):
        incidents = []
        try:
            for _ in range(min(batch_size, self.incidents.qsize())):
                incidents.append(await self.incidents.get())
        except asyncio.QueueEmpty:
            pass
        return incidents


# Example usage
async def main():
    config = {
        'endpoint': '/api/v1/incidents',
        'port': 5000,
        'rate_limit': {
            'requests': 100,
            'per_seconds': 60
        }
    }
    incident_input = IncidentInputInterface(config)
    await incident_input.start_server()

    # Simulate incoming incidents
    async def submit_incident(incident_id):
        async with aiohttp.ClientSession() as session:
            incident_data = {
                'id': f'INC-{incident_id}',
                'timestamp': '2023-07-07T12:00:00Z',
                'description': f'Test incident {incident_id}'
            }
            async with session.post('http://localhost:5000/api/v1/incidents', json=incident_data) as response:
                print(f"Submitted incident {incident_id}, status: {response.status}")

    tasks = [submit_incident(i) for i in range(1000)]
    await asyncio.gather(*tasks)

    # Process incidents
    while True:
        incidents = await incident_input.get_incidents(batch_size=10)
        if not incidents:
            break
        print(f"Processing {len(incidents)} incidents")


if __name__ == "__main__":
    asyncio.run(main())
