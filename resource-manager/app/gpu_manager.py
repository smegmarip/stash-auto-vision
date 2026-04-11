"""
Resource Manager Service - GPU Manager
Core GPU resource orchestration logic.

Uses actual hardware VRAM readings (via a callback into pynvml) as the
ground truth for available memory, rather than relying solely on lease
accounting.  An "unaccounted" bucket tracks VRAM consumed outside the
lease system (CUDA context overhead, bitsandbytes leaked memory, services
that load models without requesting leases, etc.).
"""

import asyncio
import uuid
import time
from typing import Callable, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import heapq

logger = logging.getLogger(__name__)


@dataclass
class GPULease:
    """Active GPU lease"""
    lease_id: str
    service_name: str
    vram_allocated_mb: float
    granted_at: datetime
    expires_at: datetime
    last_heartbeat: datetime
    job_id: Optional[str] = None
    perpetual: bool = False  # Perpetual leases never expire (for always-loaded models)

    def is_expired(self) -> bool:
        if self.perpetual:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "service_name": self.service_name,
            "vram_allocated_mb": self.vram_allocated_mb,
            "granted_at": self.granted_at.isoformat() + "Z",
            "expires_at": self.expires_at.isoformat() + "Z",
            "last_heartbeat": self.last_heartbeat.isoformat() + "Z",
            "job_id": self.job_id,
            "perpetual": self.perpetual,
        }


@dataclass(order=True)
class GPURequest:
    """GPU request in queue"""
    # Priority ordering (lower number = higher priority)
    priority: int
    requested_at: float = field(compare=True)  # timestamp for FIFO within priority

    # Non-comparison fields
    request_id: str = field(compare=False)
    service_name: str = field(compare=False)
    vram_required_mb: float = field(compare=False)
    timeout_at: datetime = field(compare=False)
    job_id: Optional[str] = field(compare=False, default=None)
    future: Optional[asyncio.Future] = field(compare=False, default=None)

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.timeout_at

    def to_dict(self, position: int) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_name": self.service_name,
            "vram_required_mb": self.vram_required_mb,
            "priority": self.priority,
            "requested_at": datetime.fromtimestamp(self.requested_at).isoformat() + "Z",
            "timeout_at": self.timeout_at.isoformat() + "Z",
            "position": position
        }


class GPUManager:
    """
    Manages GPU resource allocation across services

    Uses a priority queue for fair scheduling and lease-based allocation
    with heartbeat timeout for abandoned resources.
    """

    def __init__(
        self,
        total_vram_mb: float = 16384.0,  # 16GB default (RTX A4000)
        lease_duration_seconds: float = 600.0,  # 10 minutes
        heartbeat_timeout_seconds: float = 60.0,  # 1 minute without heartbeat
        cleanup_interval_seconds: float = 10.0,
        get_actual_vram: Optional[Callable[[], Tuple[float, float]]] = None,
    ):
        """
        Initialize GPU manager

        Args:
            total_vram_mb: Total available VRAM (fallback if hardware read unavailable)
            lease_duration_seconds: Default lease duration
            heartbeat_timeout_seconds: Time before lease expires without heartbeat
            cleanup_interval_seconds: Interval for cleanup task
            get_actual_vram: Optional callback returning (used_mb, total_mb) from hardware.
                When provided, available VRAM is based on actual hardware readings
                rather than lease accounting alone.
        """
        self.total_vram_mb = total_vram_mb
        self.lease_duration = timedelta(seconds=lease_duration_seconds)
        self.heartbeat_timeout = timedelta(seconds=heartbeat_timeout_seconds)
        self.cleanup_interval = cleanup_interval_seconds
        self._get_actual_vram = get_actual_vram

        # Active leases: lease_id -> GPULease
        self.active_leases: Dict[str, GPULease] = {}

        # VRAM used outside the lease system (CUDA context, leaked memory,
        # services that don't request leases, etc.)
        self.unaccounted_vram_mb: float = 0.0

        # Request queue (priority heap)
        self.request_queue: List[GPURequest] = []

        # Request lookup: request_id -> GPURequest
        self.pending_requests: Dict[str, GPURequest] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(f"GPU Manager initialized: {total_vram_mb:.0f}MB VRAM")

    async def start(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("GPU Manager cleanup task started")

    async def stop(self):
        """Stop background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("GPU Manager cleanup task stopped")

    @property
    def leased_vram_mb(self) -> float:
        """VRAM accounted for by active leases."""
        return sum(lease.vram_allocated_mb for lease in self.active_leases.values())

    @property
    def allocated_vram_mb(self) -> float:
        """Total VRAM considered in use (leases + unaccounted)."""
        return self.leased_vram_mb + self.unaccounted_vram_mb

    @property
    def available_vram_mb(self) -> float:
        """Available VRAM based on hardware readings when possible."""
        if self._get_actual_vram:
            try:
                used_mb, total_mb = self._get_actual_vram()
                if total_mb > 0:
                    return max(0, total_mb - used_mb)
            except Exception:
                pass
        return max(0, self.total_vram_mb - self.allocated_vram_mb)

    async def request_gpu(
        self,
        service_name: str,
        vram_required_mb: float,
        priority: int = 5,
        timeout_seconds: float = 300.0,
        job_id: Optional[str] = None,
        perpetual: bool = False,
    ) -> Dict[str, Any]:
        """
        Request GPU access

        Args:
            service_name: Name of requesting service
            vram_required_mb: VRAM needed
            priority: Priority level (1-10, lower is higher priority)
            timeout_seconds: Maximum wait time
            job_id: Optional job ID for tracking

        Returns:
            Dict with request_id, granted status, lease info
        """
        request_id = str(uuid.uuid4())

        async with self._lock:
            # Check if request can be immediately granted
            if vram_required_mb <= self.available_vram_mb:
                lease = await self._create_lease(
                    service_name, vram_required_mb, job_id, perpetual=perpetual
                )
                lease_type = "perpetual " if perpetual else ""
                logger.info(
                    f"GPU {lease_type}request granted immediately: {service_name} "
                    f"({vram_required_mb:.0f}MB)"
                )
                return {
                    "request_id": request_id,
                    "granted": True,
                    "lease_id": lease.lease_id,
                    "queue_position": None,
                    "estimated_wait_seconds": None,
                    "message": "GPU access granted"
                }

            # Queue the request
            timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)
            request = GPURequest(
                priority=priority,
                requested_at=time.time(),
                request_id=request_id,
                service_name=service_name,
                vram_required_mb=vram_required_mb,
                timeout_at=timeout_at,
                job_id=job_id,
                future=asyncio.get_event_loop().create_future()
            )

            heapq.heappush(self.request_queue, request)
            self.pending_requests[request_id] = request

            position = self._get_queue_position(request_id)
            estimated_wait = self._estimate_wait_time(position)

            logger.info(
                f"GPU request queued: {service_name} ({vram_required_mb:.0f}MB) "
                f"position={position}"
            )

            return {
                "request_id": request_id,
                "granted": False,
                "lease_id": None,
                "queue_position": position,
                "estimated_wait_seconds": estimated_wait,
                "message": f"Request queued at position {position}"
            }

    async def wait_for_grant(self, request_id: str) -> Dict[str, Any]:
        """
        Wait for a queued request to be granted

        Args:
            request_id: Request ID to wait for

        Returns:
            Dict with grant status and lease info
        """
        request = self.pending_requests.get(request_id)
        if not request:
            return {
                "granted": False,
                "lease_id": None,
                "message": "Request not found"
            }

        try:
            # Wait for the request to be processed
            remaining = (request.timeout_at - datetime.utcnow()).total_seconds()
            if remaining <= 0:
                return {
                    "granted": False,
                    "lease_id": None,
                    "message": "Request timed out"
                }

            lease_id = await asyncio.wait_for(request.future, timeout=remaining)
            return {
                "granted": True,
                "lease_id": lease_id,
                "message": "GPU access granted"
            }
        except asyncio.TimeoutError:
            return {
                "granted": False,
                "lease_id": None,
                "message": "Request timed out"
            }

    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a GPU request"""
        # Check if already granted (lease exists)
        for lease in self.active_leases.values():
            # Match by job_id if set
            request = self.pending_requests.get(request_id)
            if request and lease.job_id and lease.job_id == request.job_id:
                return {
                    "granted": True,
                    "lease_id": lease.lease_id,
                    "position": None,
                    "message": "GPU access granted"
                }

        # Check pending requests
        request = self.pending_requests.get(request_id)
        if not request:
            return {
                "granted": False,
                "cancelled": False,
                "failed": True,
                "position": None,
                "message": "Request not found"
            }

        if request.is_expired():
            return {
                "granted": False,
                "cancelled": False,
                "failed": True,
                "position": None,
                "message": "Request timed out"
            }

        position = self._get_queue_position(request_id)
        return {
            "granted": False,
            "cancelled": False,
            "failed": False,
            "position": position,
            "message": f"Waiting in queue at position {position}"
        }

    async def release_gpu(self, lease_id: str) -> Dict[str, Any]:
        """
        Release GPU access

        Args:
            lease_id: Lease ID to release

        Returns:
            Dict with release status
        """
        async with self._lock:
            lease = self.active_leases.pop(lease_id, None)

            if not lease:
                return {
                    "released": False,
                    "message": "Lease not found"
                }

            logger.info(
                f"GPU released: {lease.service_name} ({lease.vram_allocated_mb:.0f}MB)"
            )

            # Try to grant next request in queue
            await self._process_queue()

            return {
                "released": True,
                "message": "GPU access released"
            }

    async def heartbeat(self, lease_id: str) -> Dict[str, Any]:
        """
        Refresh lease with heartbeat

        Args:
            lease_id: Lease ID to refresh

        Returns:
            Dict with heartbeat status
        """
        async with self._lock:
            lease = self.active_leases.get(lease_id)

            if not lease:
                return {
                    "success": False,
                    "lease_valid": False,
                    "expires_in_seconds": 0,
                    "message": "Lease not found"
                }

            # Refresh lease
            now = datetime.utcnow()
            lease.last_heartbeat = now
            lease.expires_at = now + self.lease_duration

            expires_in = (lease.expires_at - now).total_seconds()

            return {
                "success": True,
                "lease_valid": True,
                "expires_in_seconds": expires_in,
                "message": "Heartbeat received"
            }

    async def _create_lease(
        self,
        service_name: str,
        vram_mb: float,
        job_id: Optional[str] = None,
        perpetual: bool = False,
    ) -> GPULease:
        """Create a new GPU lease"""
        now = datetime.utcnow()
        lease = GPULease(
            lease_id=str(uuid.uuid4()),
            service_name=service_name,
            vram_allocated_mb=vram_mb,
            granted_at=now,
            expires_at=now + self.lease_duration,
            last_heartbeat=now,
            job_id=job_id,
            perpetual=perpetual,
        )
        self.active_leases[lease.lease_id] = lease
        return lease

    async def _process_queue(self):
        """Process pending requests in queue"""
        # Remove expired requests
        self._cleanup_expired_requests()

        # Try to grant requests
        while self.request_queue:
            # Peek at highest priority request
            request = self.request_queue[0]

            if request.is_expired():
                heapq.heappop(self.request_queue)
                self.pending_requests.pop(request.request_id, None)
                continue

            if request.vram_required_mb <= self.available_vram_mb:
                # Grant this request
                heapq.heappop(self.request_queue)
                self.pending_requests.pop(request.request_id, None)

                lease = await self._create_lease(
                    request.service_name,
                    request.vram_required_mb,
                    request.job_id
                )

                logger.info(
                    f"GPU request granted from queue: {request.service_name} "
                    f"({request.vram_required_mb:.0f}MB)"
                )

                # Signal waiting coroutine
                if request.future and not request.future.done():
                    request.future.set_result(lease.lease_id)
            else:
                # Can't satisfy next request, stop processing
                break

    def _cleanup_expired_requests(self):
        """Remove expired requests from queue"""
        valid_requests = []
        while self.request_queue:
            request = heapq.heappop(self.request_queue)
            if not request.is_expired():
                valid_requests.append(request)
            else:
                self.pending_requests.pop(request.request_id, None)
                if request.future and not request.future.done():
                    request.future.set_exception(TimeoutError("Request expired"))

        for req in valid_requests:
            heapq.heappush(self.request_queue, req)

    async def _cleanup_expired_leases(self):
        """Remove expired leases, reconciling with actual VRAM usage."""
        async with self._lock:
            expired = [
                lease_id for lease_id, lease in self.active_leases.items()
                if not lease.perpetual and (
                    lease.is_expired() or
                    (datetime.utcnow() - lease.last_heartbeat) > self.heartbeat_timeout
                )
            ]

            if expired and self._get_actual_vram:
                # Snapshot VRAM before removing leases
                try:
                    used_before, _ = self._get_actual_vram()
                except Exception:
                    used_before = None

            for lease_id in expired:
                lease = self.active_leases.pop(lease_id)
                logger.warning(
                    f"Lease expired: {lease.service_name} ({lease.vram_allocated_mb:.0f}MB)"
                )

            if expired:
                await self._process_queue()

    async def _reconcile_vram(self):
        """Reconcile lease-based accounting with actual hardware VRAM usage.

        Updates the unaccounted bucket to reflect VRAM consumed outside the
        lease system (CUDA context, leaked memory, unleased services, etc.).
        """
        if not self._get_actual_vram:
            return

        try:
            actual_used, actual_total = self._get_actual_vram()
        except Exception:
            return

        if actual_total <= 0:
            return

        self.total_vram_mb = actual_total
        leased = self.leased_vram_mb
        unaccounted = max(0, actual_used - leased)

        if abs(unaccounted - self.unaccounted_vram_mb) > 50:  # 50 MB noise threshold
            logger.info(
                f"VRAM reconciliation: actual_used={actual_used:.0f}MB, "
                f"leased={leased:.0f}MB, unaccounted={unaccounted:.0f}MB "
                f"(was {self.unaccounted_vram_mb:.0f}MB)"
            )
            self.unaccounted_vram_mb = unaccounted

    async def _cleanup_loop(self):
        """Background cleanup and reconciliation task"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_leases()
                await self._reconcile_vram()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def _get_queue_position(self, request_id: str) -> int:
        """Get position of request in queue"""
        sorted_queue = sorted(self.request_queue)
        for i, req in enumerate(sorted_queue):
            if req.request_id == request_id:
                return i + 1
        return -1

    def _estimate_wait_time(self, position: int) -> float:
        """Estimate wait time based on queue position"""
        # Simple estimate: 60 seconds per position
        return position * 60.0

    def get_status(self) -> Dict[str, Any]:
        """Get current GPU status with hardware-reconciled VRAM info."""
        actual_used: Optional[float] = None
        actual_total: Optional[float] = None
        if self._get_actual_vram:
            try:
                actual_used, actual_total = self._get_actual_vram()
            except Exception:
                pass

        return {
            "status": "in_use" if self.active_leases else "available",
            "total_vram_mb": actual_total or self.total_vram_mb,
            "available_vram_mb": self.available_vram_mb,
            "allocated_vram_mb": self.allocated_vram_mb,
            "leased_vram_mb": self.leased_vram_mb,
            "unaccounted_vram_mb": round(self.unaccounted_vram_mb, 1),
            "actual_used_vram_mb": round(actual_used, 1) if actual_used is not None else None,
            "active_leases": [lease.to_dict() for lease in self.active_leases.values()],
            "queue_length": len(self.request_queue),
            "queue": [
                req.to_dict(i + 1)
                for i, req in enumerate(sorted(self.request_queue))
            ]
        }
