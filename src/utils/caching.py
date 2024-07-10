import functools
import asyncio
import aioredis
import json
import logging

logger = logging.getLogger(__name__)

redis_client = None

async def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = await aioredis.from_url("redis://localhost")
    return redis_client

def cache(ttl=3600):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            redis = await get_redis_client()

            # Try to get the result from cache
            cached_result = await redis.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key}")
                return json.loads(cached_result)

            # If not in cache, call the function
            result = await func(*args, **kwargs)

            # Store the result in cache
            await redis.set(cache_key, json.dumps(result), ex=ttl)
            logger.info(f"Cached result for {cache_key}")

            return result
        return wrapper
    return decorator

async def clear_cache(pattern="*"):
    redis = await get_redis_client()
    keys = await redis.keys(pattern)
    if keys:
        await redis.delete(*keys)
        logger.info(f"Cleared {len(keys)} keys from cache")
    else:
        logger.info("No keys found in cache to clear")