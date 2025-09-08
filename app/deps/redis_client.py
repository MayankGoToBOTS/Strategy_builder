import redis.asyncio as redis
from redis.asyncio import Redis
from app.core.config import settings
import logging
import json
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class RedisManager:
    pool: Optional[redis.ConnectionPool] = None
    redis: Optional[Redis] = None

    @classmethod
    async def initialize(cls):
        """Initialize Redis connection"""
        try:
            cls.pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                db=settings.REDIS_DB,
                max_connections=20,
                retry_on_timeout=True,
                decode_responses=True
            )
            cls.redis = Redis(connection_pool=cls.pool)
            
            # Test the connection
            await cls.redis.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    @classmethod
    async def close(cls):
        """Close Redis connection"""
        if cls.redis:
            # Fixed: Use aclose() instead of close() to avoid deprecation warning
            await cls.redis.aclose()
        if cls.pool:
            await cls.pool.disconnect()
        logger.info("Redis connection closed")

    @classmethod
    async def health_check(cls) -> str:
        """Check Redis health"""
        try:
            await cls.redis.ping()
            return "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return "unhealthy"

    @classmethod
    def get_redis(cls) -> Redis:
        """Get Redis instance"""
        if cls.redis is None:
            raise RuntimeError("Redis not initialized. Call initialize() first.")
        return cls.redis

    # Helper methods for common Redis operations
    @classmethod
    async def set_json(cls, key: str, value: Dict[Any, Any], ex: Optional[int] = None):
        """Set JSON value in Redis"""
        await cls.redis.set(key, json.dumps(value), ex=ex)

    @classmethod
    async def get_json(cls, key: str) -> Optional[Dict[Any, Any]]:
        """Get JSON value from Redis"""
        value = await cls.redis.get(key)
        if value:
            return json.loads(value)
        return None

    @classmethod
    async def set_hash(cls, key: str, mapping: Dict[str, Any]):
        """Set hash fields in Redis"""
        await cls.redis.hset(key, mapping=mapping)

    @classmethod
    async def get_hash(cls, key: str) -> Dict[str, str]:
        """Get all hash fields from Redis"""
        return await cls.redis.hgetall(key)

    @classmethod
    async def publish_stream(cls, stream: str, data: Dict[str, Any]) -> str:
        """Publish data to Redis stream"""
        return await cls.redis.xadd(stream, data)

    @classmethod
    async def read_stream(cls, stream: str, count: int = 10, block: Optional[int] = None) -> List:
        """Read from Redis stream"""
        return await cls.redis.xread({stream: '$'}, count=count, block=block)

# Dependency function for FastAPI
async def get_redis() -> Redis:
    return RedisManager.get_redis()