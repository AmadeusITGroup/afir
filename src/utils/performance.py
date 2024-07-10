import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


async def batch_process(items, process_func, batch_size=100, max_workers=4):
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_results = await asyncio.gather(*[
                asyncio.to_thread(process_func, item) for item in batch
            ])
        results.extend(batch_results)
        logger.info(f"Processed batch of {len(batch)} items")
    return results


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
async def example_usage():
    async def process_item(item):
        await asyncio.sleep(0.1)  # Simulate some processing
        return item * 2

    items = list(range(1000))
    results = await batch_process(items, process_item)
    print(f"Processed {len(results)} items")

    rate_limiter = AsyncRateLimiter(rate_limit=10, time_period=1)
    for _ in range(20):
        await rate_limiter.acquire()
        print("Acquired token")
    rate_limiter.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
