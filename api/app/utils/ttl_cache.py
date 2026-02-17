from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class AsyncTTLCache(Generic[T]):
    """
    Tiny in-memory TTL cache for FastAPI. Single-process cache.
    Safe for concurrent requests via an asyncio.Lock.
    """
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._entry: Optional[CacheEntry[T]] = None
        self._lock = asyncio.Lock()

    def _is_valid(self) -> bool:
        return self._entry is not None and time.time() < self._entry.expires_at

    async def get(self) -> Optional[T]:
        async with self._lock:
            if self._is_valid():
                return self._entry.value
            return None

    async def set(self, value: T) -> None:
        async with self._lock:
            self._entry = CacheEntry(value=value, expires_at=time.time() + self.ttl_seconds)

    async def clear(self) -> None:
        async with self._lock:
            self._entry = None
