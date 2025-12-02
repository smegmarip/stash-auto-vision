"""
Schema Service - Cache Manager
In-memory caching with background refresh
"""

import asyncio
from time import time
from typing import Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple in-memory cache with optional background refresh"""

    def __init__(self, providerFn: Callable[[], Any], refresh_interval: Optional[int] = 300):
        """
        Initialize cache manager
        Args:
            providerFn: Async function to provide fresh data
            refresh_interval: Background refresh interval in seconds (optional, default 300s)
        """
        self.providerFn = providerFn
        self.scheduled_task: Optional[asyncio.Task] = None
        self.refresh_interval = refresh_interval
        self.cache = None
        self.last_run = 0

        if not self.providerFn or not asyncio.iscoroutinefunction(self.providerFn):
            raise ValueError("providerFn must be an async function")

    def start(self):
        """Start background refresh task if interval is set."""

        if self.refresh_interval and self.refresh_interval > 0:
            self.scheduled_task = asyncio.create_task(self.background())
            return True
        return False

    async def background(self):
        """Background loop to refresh cache periodically."""
        while True:
            try:
                self.cache = await self.providerFn()
                self.last_run = time()
            except Exception as e:
                logger.error(f"Cache background loop error: {e}")

            await asyncio.sleep(self.refresh_interval)

    async def refresh(self):
        """One-off non-blocking refresh when TTL expires."""
        try:
            self.cache = await self.providerFn()
            self.last_run = time()
        except Exception as e:
            logger.error(f"Cache refresh error: {e}")

    def get(self) -> Any:
        """Get current cache value."""
        if self.scheduled_task is None and self.refresh_interval and time() - self.last_run > self.refresh_interval:
            asyncio.create_task(self.refresh())
        return self.cache

    async def stop(self):
        """Cleanup resources."""
        if self.scheduled_task:
            self.scheduled_task.cancel()
            try:
                await self.scheduled_task
                self.scheduled_task = None
            except asyncio.CancelledError:
                pass
