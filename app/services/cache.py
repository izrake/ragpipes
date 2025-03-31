import redis
import json
from typing import Optional, Dict, Any, List
from app.core.config import get_settings
from datetime import datetime

settings = get_settings()

class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        self.cache_ttl = 3600  # 1 hour in seconds

    async def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for a query"""
        cached_data = self.redis_client.get(f"query:{query}")
        if cached_data:
            return json.loads(cached_data)
        return None

    async def cache_response(
        self,
        query: str,
        response: str,
        context_chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Cache a query response"""
        cache_data = {
            "response": response,
            "context_chunks": context_chunks,
            "metadata": metadata or {},
            "timestamp": str(datetime.now())
        }
        self.redis_client.setex(
            f"query:{query}",
            self.cache_ttl,
            json.dumps(cache_data)
        )

    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for a text"""
        cached_data = self.redis_client.get(f"embedding:{text}")
        if cached_data:
            return json.loads(cached_data)
        return None

    async def cache_embedding(self, text: str, embedding: List[float]):
        """Cache an embedding"""
        self.redis_client.setex(
            f"embedding:{text}",
            self.cache_ttl,
            json.dumps(embedding)
        )

    async def invalidate_cache(self, pattern: str = "*"):
        """Invalidate cache entries matching a pattern"""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys) 