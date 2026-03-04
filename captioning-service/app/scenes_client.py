"""
Captioning Service - Scenes Service Client
Client for fetching scene boundaries from scenes-service
"""

from typing import List, Optional, Dict, Any
import httpx
import logging

logger = logging.getLogger(__name__)


class ScenesServerClient:
    """Client for scenes-service"""

    def __init__(
        self,
        scenes_server_url: str = "http://scenes-service:5002",
        timeout: float = 30.0
    ):
        """
        Initialize scenes server client

        Args:
            scenes_server_url: URL of scenes-service
            timeout: Request timeout in seconds
        """
        self.scenes_server_url = scenes_server_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_scene_boundaries(
        self,
        job_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get scene boundaries from a completed scenes job

        Args:
            job_id: Scenes job ID

        Returns:
            List of scene boundary dicts with start_timestamp, end_timestamp
        """
        client = await self._get_client()

        try:
            # First check job status
            status_response = await client.get(
                f"{self.scenes_server_url}/scenes/jobs/{job_id}/status"
            )
            status_response.raise_for_status()
            status = status_response.json()

            if status.get("status") != "completed":
                raise Exception(f"Scenes job not completed: {status.get('status')}")

            # Get results
            results_response = await client.get(
                f"{self.scenes_server_url}/scenes/jobs/{job_id}/results"
            )
            results_response.raise_for_status()
            results = results_response.json()

            # Extract boundaries - scenes is an array directly
            scenes = results.get("scenes", [])

            boundaries = []
            for scene in scenes:
                boundaries.append({
                    "scene_index": scene.get("scene_number", len(boundaries)),
                    "start_timestamp": scene.get("start_timestamp", 0.0),
                    "end_timestamp": scene.get("end_timestamp", 0.0),
                    "start_frame": scene.get("start_frame"),
                    "end_frame": scene.get("end_frame")
                })

            logger.info(f"Retrieved {len(boundaries)} scene boundaries from job {job_id}")
            return boundaries

        except Exception as e:
            logger.error(f"Error getting scene boundaries: {e}")
            raise

    async def analyze_scenes(
        self,
        video_path: str,
        threshold: float = 27.0,
        poll_interval: float = 1.0,
        timeout: float = 300.0
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Trigger scene detection and wait for completion.

        Args:
            video_path: Path to video file
            threshold: Scene detection threshold
            poll_interval: Seconds between status polls
            timeout: Maximum wait time in seconds

        Returns:
            Tuple of (job_id, scene_boundaries)
        """
        import asyncio
        import time

        client = await self._get_client()
        start_time = time.time()

        try:
            # Submit scene detection job
            response = await client.post(
                f"{self.scenes_server_url}/scenes/detect",
                json={
                    "video_path": video_path,
                    "scene_threshold": threshold
                }
            )
            response.raise_for_status()
            job_info = response.json()
            job_id = job_info.get("job_id")

            if not job_id:
                raise RuntimeError("No job_id returned from scenes service")

            logger.info(f"Scene detection job submitted: {job_id}")

            # Check for cache hit
            if job_info.get("status") == "completed" and job_info.get("cache_hit"):
                logger.info(f"Scene detection cache hit: {job_id}")
                boundaries = await self.get_scene_boundaries(job_id)
                return job_id, boundaries

            # Poll for completion
            while time.time() - start_time < timeout:
                status_response = await client.get(
                    f"{self.scenes_server_url}/scenes/jobs/{job_id}/status"
                )
                status = status_response.json()

                if status.get("status") == "completed":
                    logger.info(f"Scene detection completed: {job_id}")
                    boundaries = await self.get_scene_boundaries(job_id)
                    return job_id, boundaries
                elif status.get("status") == "failed":
                    error_msg = status.get("error", "Unknown error")
                    raise RuntimeError(f"Scene detection failed: {error_msg}")

                await asyncio.sleep(poll_interval)

            raise RuntimeError(f"Scene detection timed out after {timeout}s")

        except Exception as e:
            logger.error(f"Error during scene detection: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if scenes service is reachable"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.scenes_server_url}/scenes/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Scenes server health check failed: {e}")
            return False
