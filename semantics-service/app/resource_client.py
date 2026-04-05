"""
Captioning Service - Resource Manager Client
Client for communicating with resource-manager service for GPU orchestration
"""

import asyncio
from typing import Optional, Dict, Any
import httpx
import logging

logger = logging.getLogger(__name__)


class ResourceManagerClient:
    """Client for resource-manager service"""

    def __init__(
        self,
        resource_manager_url: str = "http://resource-manager:5007",
        service_name: str = "captioning-service",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize resource manager client

        Args:
            resource_manager_url: URL of resource-manager service
            service_name: Name of this service for registration
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for requests
        """
        self.resource_manager_url = resource_manager_url.rstrip("/")
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._gpu_lease_id: Optional[str] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
        return self._client

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        retry: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request with retries"""
        client = await self._get_client()
        url = f"{self.resource_manager_url}{endpoint}"

        attempts = self.max_retries if retry else 1

        for attempt in range(attempts):
            try:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=json)
                elif method == "DELETE":
                    response = await client.delete(url)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if attempt < attempts - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise
            except Exception as e:
                if attempt < attempts - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                logger.error(f"Request failed: {method} {url}: {e}")
                raise

    async def request_gpu(
        self,
        vram_mb: float,
        priority: int = 5,
        timeout_seconds: float = 300.0
    ) -> Dict[str, Any]:
        """
        Request GPU access from resource manager

        Args:
            vram_mb: Required VRAM in MB
            priority: Priority level (1=highest, 10=lowest)
            timeout_seconds: Maximum wait time for GPU access

        Returns:
            Dict with lease_id, granted status, and wait info
        """
        logger.info(f"Requesting GPU access: {vram_mb:.0f}MB VRAM, priority={priority}")

        response = await self._request(
            "POST",
            "/resources/gpu/request",
            json={
                "service_name": self.service_name,
                "vram_required_mb": vram_mb,
                "priority": priority,
                "timeout_seconds": timeout_seconds
            }
        )

        if response.get("granted"):
            self._gpu_lease_id = response.get("lease_id")
            logger.info(f"GPU access granted, lease_id={self._gpu_lease_id}")
        else:
            logger.info(f"GPU request queued, position={response.get('queue_position')}")

        return response

    async def wait_for_gpu(
        self,
        request_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 300.0
    ) -> Dict[str, Any]:
        """
        Wait for GPU access to be granted

        Args:
            request_id: Request ID from initial request
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait

        Returns:
            Dict with granted status and lease info
        """
        elapsed = 0.0

        while elapsed < max_wait:
            status = await self.get_request_status(request_id)

            if status.get("granted"):
                self._gpu_lease_id = status.get("lease_id")
                logger.info(f"GPU access granted after {elapsed:.1f}s")
                return status

            if status.get("cancelled") or status.get("failed"):
                raise Exception(f"GPU request failed: {status.get('message')}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"GPU access not granted within {max_wait}s")

    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a GPU request"""
        return await self._request("GET", f"/resources/gpu/request/{request_id}")

    async def release_gpu(self, lease_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Release GPU access

        Args:
            lease_id: Optional lease ID (uses stored ID if not provided)

        Returns:
            Release confirmation
        """
        lease_id = lease_id or self._gpu_lease_id

        if not lease_id:
            logger.warning("No GPU lease to release")
            return {"released": False, "message": "No active lease"}

        logger.info(f"Releasing GPU lease: {lease_id}")

        response = await self._request(
            "POST",
            "/resources/gpu/release",
            json={"lease_id": lease_id}
        )

        self._gpu_lease_id = None
        return response

    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU resource status"""
        return await self._request("GET", "/resources/gpu/status")

    async def heartbeat(self, lease_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send heartbeat to keep GPU lease alive

        Args:
            lease_id: Optional lease ID (uses stored ID if not provided)

        Returns:
            Heartbeat response
        """
        lease_id = lease_id or self._gpu_lease_id

        if not lease_id:
            return {"success": False, "message": "No active lease"}

        return await self._request(
            "POST",
            "/resources/gpu/heartbeat",
            json={"lease_id": lease_id}
        )

    async def health_check(self) -> bool:
        """Check if resource manager is reachable"""
        try:
            response = await self._request("GET", "/resources/health", retry=False)
            return response.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Resource manager health check failed: {e}")
            return False

    @property
    def has_gpu_lease(self) -> bool:
        """Check if we currently have a GPU lease"""
        return self._gpu_lease_id is not None

    @property
    def current_lease_id(self) -> Optional[str]:
        """Get current GPU lease ID"""
        return self._gpu_lease_id
