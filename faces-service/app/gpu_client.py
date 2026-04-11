"""
GPU Client — local lease and state manager for GPU-consuming services.

Each service that uses GPU resources gets its own GPUClient instance.
The client manages lease lifecycle, busy/idle state tracking, and
provides the interface that the resource-manager uses for cooperative
eviction.

Lease lifecycle:
  1. Service calls gpu_client.lease(vram_mb, ...) before loading models
  2. Service subscribes callers to the event handler to report busy/idle
  3. While a job is running, the lease is marked busy
  4. When the job finishes, the lease is marked idle
  5. The resource-manager may call the service's /resources/{lease_id}/release
     endpoint to evict an idle lease
  6. On eviction, gpu_client.evict(lease_id) triggers cleanup callbacks
  7. Expired leases stay in memory but cannot be reused without re-leasing
"""

import asyncio
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LeaseState(str, Enum):
    """Lease lifecycle states."""
    PENDING = "pending"       # Lease requested, not yet granted
    ACTIVE = "active"         # Lease granted, models may or may not be loaded
    BUSY = "busy"             # Lease in use (job actively running)
    IDLE = "idle"             # Lease granted, models loaded, no job running
    RELEASING = "releasing"   # Eviction in progress
    RELEASED = "released"     # VRAM freed, lease no longer valid
    EXPIRED = "expired"       # Lease expired by TTL


@dataclass
class LeaseRecord:
    """Local tracking for a single GPU lease."""
    lease_id: str
    vram_mb: float
    state: LeaseState
    perpetual: bool
    granted_at: float          # monotonic clock
    expires_at: Optional[float]  # None for perpetual
    last_heartbeat: float

    def is_expired(self) -> bool:
        if self.perpetual:
            return False
        if self.expires_at is None:
            return False
        return time.monotonic() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "vram_mb": self.vram_mb,
            "state": self.state.value,
            "perpetual": self.perpetual,
        }


# Type for eviction callbacks: async fn(lease_id) -> bool (success)
EvictCallback = Callable[[str], Awaitable[bool]]


class GPUClient:
    """Local GPU lease manager for a single service.

    Communicates with the resource-manager HTTP API for lease acquisition
    and heartbeats. Tracks lease state locally and provides the interface
    for the resource-manager's revocation protocol.
    """

    def __init__(
        self,
        resource_manager_url: str = "http://resource-manager:5007",
        service_name: str = "unknown-service",
        service_url: str = "",
        heartbeat_interval: float = 30.0,
        lease_ttl: float = 600.0,
    ):
        self.resource_manager_url = resource_manager_url.rstrip("/")
        self.service_name = service_name
        self.service_url = service_url
        self.heartbeat_interval = heartbeat_interval
        self.lease_ttl = lease_ttl

        # Active leases: lease_id -> LeaseRecord
        self._leases: Dict[str, LeaseRecord] = {}

        # Eviction callbacks: lease_id -> callback
        self._evict_callbacks: Dict[str, EvictCallback] = {}

        # HTTP client (lazy init)
        self._client: Optional[httpx.AsyncClient] = None

        # Background heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0, headers={"Content-Type": "application/json"})
        return self._client

    async def close(self):
        """Shutdown: cancel heartbeat, close HTTP client."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    # ----- Lease acquisition -----

    async def lease(
        self,
        vram_mb: float,
        priority: int = 5,
        perpetual: bool = False,
        timeout_seconds: float = 600.0,
    ) -> Optional[str]:
        """Request a GPU lease from the resource manager.

        Args:
            vram_mb: VRAM to reserve.
            priority: 1 (highest) to 10 (lowest).
            perpetual: If True, lease never expires.
            timeout_seconds: Max wait if queued.

        Returns:
            lease_id if granted, None if failed.
        """
        client = await self._get_client()
        url = f"{self.resource_manager_url}/resources/gpu/request"

        try:
            resp = await client.post(url, json={
                "service_name": self.service_name,
                "vram_required_mb": vram_mb,
                "priority": priority,
                "timeout_seconds": timeout_seconds,
                "perpetual": perpetual,
                "callback_url": self.service_url,
            })
            resp.raise_for_status()
            data = resp.json()

            if data.get("granted"):
                lease_id = data["lease_id"]
                self._register_lease(lease_id, vram_mb, perpetual)
                logger.info(f"GPU lease granted: {lease_id} ({vram_mb:.0f} MB, {'perpetual' if perpetual else 'standard'})")
                self._ensure_heartbeat()
                return lease_id

            # Queued — poll until granted or timeout
            request_id = data.get("request_id")
            if request_id:
                lease_id = await self._wait_for_grant(request_id, timeout_seconds)
                if lease_id:
                    self._register_lease(lease_id, vram_mb, perpetual)
                    logger.info(f"GPU lease granted after wait: {lease_id} ({vram_mb:.0f} MB)")
                    self._ensure_heartbeat()
                    return lease_id

        except Exception as e:
            logger.warning(f"GPU lease request failed: {e}")

        return None

    async def _wait_for_grant(self, request_id: str, max_wait: float) -> Optional[str]:
        """Poll resource manager until request is granted or times out."""
        client = await self._get_client()
        url = f"{self.resource_manager_url}/resources/gpu/request/{request_id}"
        elapsed = 0.0
        poll_interval = 2.0

        while elapsed < max_wait:
            try:
                resp = await client.get(url)
                data = resp.json()
                if data.get("granted"):
                    return data.get("lease_id")
                if data.get("failed"):
                    logger.warning(f"GPU request failed: {data.get('message')}")
                    return None
            except Exception as e:
                logger.debug(f"Poll error: {e}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(f"GPU lease request timed out after {max_wait}s")
        return None

    def _register_lease(self, lease_id: str, vram_mb: float, perpetual: bool):
        """Record a newly granted lease."""
        now = time.monotonic()
        self._leases[lease_id] = LeaseRecord(
            lease_id=lease_id,
            vram_mb=vram_mb,
            state=LeaseState.ACTIVE,
            perpetual=perpetual,
            granted_at=now,
            expires_at=None if perpetual else now + self.lease_ttl,
            last_heartbeat=now,
        )

    # ----- Busy/idle state management -----

    def mark_busy(self, lease_id: str):
        """Mark a lease as busy (job actively using GPU)."""
        rec = self._leases.get(lease_id)
        if rec and rec.state in (LeaseState.ACTIVE, LeaseState.IDLE):
            rec.state = LeaseState.BUSY
            logger.debug(f"Lease {lease_id} marked busy")

    def mark_idle(self, lease_id: str):
        """Mark a lease as idle (models loaded but no job running)."""
        rec = self._leases.get(lease_id)
        if rec and rec.state == LeaseState.BUSY:
            rec.state = LeaseState.IDLE
            logger.debug(f"Lease {lease_id} marked idle")

    def mark_released(self, lease_id: str):
        """Mark a lease as released (VRAM freed)."""
        rec = self._leases.get(lease_id)
        if rec:
            rec.state = LeaseState.RELEASED
            logger.debug(f"Lease {lease_id} marked released")

    def is_busy(self, lease_id: str) -> bool:
        """Check if a lease is currently busy."""
        rec = self._leases.get(lease_id)
        if not rec:
            return False
        return rec.state == LeaseState.BUSY

    def get_lease(self, lease_id: str) -> Optional[LeaseRecord]:
        """Get a lease record by ID."""
        return self._leases.get(lease_id)

    def get_all_leases(self) -> Dict[str, LeaseRecord]:
        """Get all lease records."""
        return dict(self._leases)

    # ----- Eviction -----

    def on_evict(self, lease_id: str, callback: EvictCallback):
        """Register an async callback to handle eviction for a lease.

        The callback receives the lease_id and should free GPU resources.
        It must return True if eviction succeeded, False otherwise.
        """
        self._evict_callbacks[lease_id] = callback

    async def evict(self, lease_id: str) -> bool:
        """Evict a lease: trigger the cleanup callback and free VRAM.

        Returns True if eviction succeeded. Called by the service's
        /resources/{lease_id}/release endpoint.
        """
        rec = self._leases.get(lease_id)
        if not rec:
            logger.warning(f"Cannot evict unknown lease: {lease_id}")
            return False

        if rec.state == LeaseState.BUSY:
            logger.warning(f"Cannot evict busy lease: {lease_id}")
            return False

        if rec.perpetual:
            logger.warning(f"Cannot evict perpetual lease: {lease_id}")
            return False

        rec.state = LeaseState.RELEASING
        logger.info(f"Evicting lease {lease_id} ({rec.vram_mb:.0f} MB)")

        callback = self._evict_callbacks.get(lease_id)
        if callback:
            try:
                success = await callback(lease_id)
                if success:
                    rec.state = LeaseState.RELEASED
                    await self._release_on_manager(lease_id)
                    logger.info(f"Lease {lease_id} evicted successfully")
                    return True
                else:
                    rec.state = LeaseState.IDLE
                    logger.warning(f"Eviction callback failed for lease {lease_id}")
                    return False
            except Exception as e:
                rec.state = LeaseState.IDLE
                logger.error(f"Eviction callback error for lease {lease_id}: {e}")
                return False
        else:
            # No callback registered — just mark released
            rec.state = LeaseState.RELEASED
            await self._release_on_manager(lease_id)
            return True

    async def release(self, lease_id: str):
        """Voluntarily release a lease (service-initiated, not eviction)."""
        rec = self._leases.get(lease_id)
        if rec:
            rec.state = LeaseState.RELEASED
        await self._release_on_manager(lease_id)
        logger.info(f"Lease {lease_id} released voluntarily")

    async def _release_on_manager(self, lease_id: str):
        """Notify the resource manager that a lease is released."""
        try:
            client = await self._get_client()
            await client.post(
                f"{self.resource_manager_url}/resources/gpu/release",
                json={"lease_id": lease_id},
            )
        except Exception as e:
            logger.debug(f"Failed to notify resource manager of release: {e}")

    # ----- Heartbeat -----

    def _ensure_heartbeat(self):
        """Start the heartbeat loop if not already running."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """Periodically heartbeat all active non-perpetual leases and check expiration."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                active_leases = [
                    r for r in self._leases.values()
                    if r.state in (LeaseState.ACTIVE, LeaseState.BUSY, LeaseState.IDLE)
                    and not r.perpetual
                ]
                if not active_leases:
                    continue

                client = await self._get_client()
                for rec in active_leases:
                    if rec.is_expired():
                        rec.state = LeaseState.EXPIRED
                        logger.info(f"Lease {rec.lease_id} expired locally")
                        continue
                    try:
                        success = False
                        for attempt in range(2):  # 1 retry
                            resp = await client.post(
                                f"{self.resource_manager_url}/resources/gpu/heartbeat",
                                json={"lease_id": rec.lease_id},
                            )
                            data = resp.json()
                            if data.get("success"):
                                rec.last_heartbeat = time.monotonic()
                                if rec.expires_at is not None:
                                    rec.expires_at = time.monotonic() + self.lease_ttl
                                success = True
                                break
                            if attempt == 0:
                                logger.debug(f"Heartbeat missed for {rec.lease_id}, retrying in 2s")
                                await asyncio.sleep(2.0)
                        if not success:
                            logger.warning(f"Heartbeat failed for lease {rec.lease_id} after retry: {data.get('message')}")
                            rec.state = LeaseState.EXPIRED
                    except Exception as e:
                        logger.debug(f"Heartbeat error for {rec.lease_id}: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
