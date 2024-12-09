import asyncio
import json
import logging
from asyncio import Queue

import requests
from aiohttp import web
from requests.auth import HTTPBasicAuth

from src.utils.error_handling import async_retry_with_backoff
from src.utils.performance import AsyncRateLimiter

logger = logging.getLogger(__name__)


class IncidentInputInterface:
    def __init__(self, config):
        self.config = config
        self.app = web.Application()
        self.app.router.add_post(config['post_incident_endpoint'], self.receive_incident)
        self.app.router.add_post(config['post_ir_endpoint'], self.get_ir)
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

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def get_ir(self, request):
        await self.rate_limiter.acquire()
        try:
            incident = await request.json()
            if not self.validate_ir_request(incident):
                return web.Response(status=400, text="Invalid IR data")

            url = self.config['win_url'] + '/api/v2/json/records/?rnid=' + incident['id'] + '&with=full'
            response = requests.get(url, auth=HTTPBasicAuth(self.config['win_username'], self.config['win_password']))

            if response.status_code == 200:
                incident = response.json()
                ir = {'id': incident['records'][0]['ffts'][0]['rnid'],
                      'timestamp': incident['records'][0]['ffts'][0]['update_date'],
                      'description': incident['records'][0]['ffts'][0]['fft']}
                await self.incidents.put(ir)
                logger.info(f"Received incident: {ir['id']}")
                return web.Response(status=201, text=f"Received incident: {ir['id']}")
            else:
                return web.Response(status=400, text=f"Could not get IR")
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error(f"Error receiving incident: {str(e)}")
            return web.Response(status=400, text="Invalid IR")

    def validate_ir_request(self, request):
        required_fields = ['id']
        return all(field in request for field in required_fields)

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
    return


if __name__ == "__main__":
    asyncio.run(main())
