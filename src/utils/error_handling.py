import asyncio
import functools
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def retry_with_backoff(max_attempts, backoff_in_seconds):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_in_seconds, min=backoff_in_seconds, max=backoff_in_seconds * 10)
    )


def async_retry_with_backoff(max_attempts, backoff_in_seconds):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Max retries reached for {func.__name__}. Error: {str(e)}")
                        raise
                    wait_time = backoff_in_seconds * (2 ** attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1} for {func.__name__}. Waiting {wait_time} seconds. Error: {str(e)}")
                    await asyncio.sleep(wait_time)

        return wrapper

    return decorator
