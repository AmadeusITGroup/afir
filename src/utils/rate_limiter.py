import asyncio


class AsyncRateLimiter:
    """Token-bucket request-rate limiter shared by ``LLMClient`` and the HTTP interface.

    The refill task is created on the running loop, so the limiter must be built from
    inside one. A module-level instance would bind to whichever loop existed at import
    time rather than the one acquiring from it.
    """

    def __init__(self, rate_limit, time_period=60):
        self.rate_limit = rate_limit
        self.time_period = time_period
        self.tokens = asyncio.Semaphore(rate_limit)
        self.task = asyncio.create_task(self.add_tokens())

    async def add_tokens(self):
        while True:
            await asyncio.sleep(self.time_period / self.rate_limit)
            self.tokens.release()

    async def acquire(self):
        await self.tokens.acquire()

    def close(self):
        self.task.cancel()
