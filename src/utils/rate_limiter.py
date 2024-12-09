import time
import asyncio
from collections import deque


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    async def __aenter__(self):
        while len(self.calls) >= self.max_calls:
            limit = self.calls[-1] + self.period
            now = time.time()
            if limit > now:
                await asyncio.sleep(limit - now)
            else:
                self.calls.popleft()
        self.calls.append(time.time())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class AsyncRateLimiter:
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


# Example usage
async def main():
    rate_limiter = AsyncRateLimiter(rate_limit=5, time_period=10)  # 5 requests per 10 seconds

    async def make_request(i):
        await rate_limiter.acquire()
        print(f"Request {i} at {time.time():.2f}")

    tasks = [asyncio.create_task(make_request(i)) for i in range(20)]
    await asyncio.gather(*tasks)

    rate_limiter.close()


if __name__ == "__main__":
    asyncio.run(main())
