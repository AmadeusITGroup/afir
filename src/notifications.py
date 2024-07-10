import asyncio
import aiohttp
from aiohttp import web
import logging

logger = logging.getLogger(__name__)


class NotificationSystem:
    def __init__(self):
        self.clients = set()

    async def send_notification(self, incident_id, status, message):
        if not self.clients:
            logger.warning("No connected clients to send notification")
            return

        notification = {
            'incident_id': incident_id,
            'status': status,
            'message': message
        }

        for ws in self.clients:
            try:
                await ws.send_json(notification)
            except Exception as e:
                logger.error(f"Failed to send notification to a client: {str(e)}")
                self.clients.remove(ws)

    async def ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.clients.add(ws)
        logger.info(f"New client connected. Total clients: {len(self.clients)}")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception {ws.exception()}")
        finally:
            self.clients.remove(ws)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

        return ws


notification_system = NotificationSystem()


def send_notification(incident_id, status, message):
    asyncio.create_task(notification_system.send_notification(incident_id, status, message))


# Example of how to set up the WebSocket server
async def start_notification_server():
    app = web.Application()
    app.router.add_get('/ws', notification_system.ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    logger.info("Notification server started on ws://localhost:8080/ws")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    async def main():
        await start_notification_server()

        # Simulate sending notifications
        while True:
            await asyncio.sleep(5)
            send_notification('INC-12345', 'processing', 'Analyzing logs...')


    asyncio.run(main())
