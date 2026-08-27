import asyncio
import functools
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def retry_with_backoff(max_attempts, backoff_in_seconds):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=backoff_in_seconds,
            min=backoff_in_seconds,
            max=backoff_in_seconds * 10,
        ),
    )


class NonRetryableError(Exception):
    """An error that retrying cannot fix. Raise it to skip the remaining attempts.

    A statement that exhausted its wall-clock budget is the case this exists for:
    retrying re-runs the same expensive query from scratch, so the next attempt blows the
    per-source cap as well and a slow source ends up reported as an empty one. Attempts
    are for transient faults such as a dropped connection or a 5xx, not for work that is
    genuinely too slow.
    """


def async_retry_with_backoff(max_attempts, backoff_in_seconds):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except NonRetryableError as e:
                    logger.error(
                        f"Not retrying {func.__name__} (non-retryable). Error: {str(e)}"
                    )
                    raise
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Max retries reached for {func.__name__}. Error: {str(e)}"
                        )
                        raise
                    wait_time = backoff_in_seconds * (2**attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1} for {func.__name__}. Waiting {wait_time} seconds. Error: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)

        return wrapper

    return decorator
