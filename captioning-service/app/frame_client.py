"""
Captioning Service - Frame Server Client
Client for fetching frames from frame-server service
"""

import asyncio
from typing import List, Optional, Dict, Any
import httpx
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class FrameServerClient:
    """Client for frame-server service"""

    def __init__(
        self,
        frame_server_url: str = "http://frame-server:5001",
        timeout: float = 120.0
    ):
        """
        Initialize frame server client

        Args:
            frame_server_url: URL of frame-server service
            timeout: Request timeout in seconds
        """
        self.frame_server_url = frame_server_url.rstrip("/")
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

    async def extract_frames(
        self,
        video_path: str,
        sampling_interval: float = 5.0,
        scene_boundaries: Optional[List[Dict]] = None,
        frames_per_scene: int = 3,
        poll_interval: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Extract frames from video via frame-server

        Args:
            video_path: Path to video file
            sampling_interval: Interval between frames in seconds (interval mode)
            scene_boundaries: Optional scene boundaries for scene-based extraction
            frames_per_scene: Frames to extract per scene
            poll_interval: Seconds between status polls

        Returns:
            Frame extraction result with metadata
        """
        client = await self._get_client()

        try:
            # If scene boundaries provided, extract specific timestamps
            if scene_boundaries:
                timestamps = self._calculate_scene_timestamps(
                    scene_boundaries, frames_per_scene
                )
                response = await client.post(
                    f"{self.frame_server_url}/frames/extract",
                    json={
                        "video_path": video_path,
                        "timestamps": timestamps
                    }
                )
            else:
                # Interval-based extraction
                response = await client.post(
                    f"{self.frame_server_url}/frames/extract",
                    json={
                        "video_path": video_path,
                        "sampling_interval": sampling_interval
                    }
                )

            response.raise_for_status()
            job_info = response.json()
            job_id = job_info.get("job_id")

            if not job_id:
                # Synchronous response (unlikely but handle it)
                return job_info

            logger.info(f"Frame extraction job submitted: {job_id}")

            # Poll for completion
            while True:
                status_response = await client.get(
                    f"{self.frame_server_url}/frames/jobs/{job_id}/status"
                )
                status = status_response.json()

                if status.get("status") == "completed":
                    logger.info(f"Frame extraction completed: {job_id}")
                    break
                elif status.get("status") == "failed":
                    error_msg = status.get("error", "Unknown error")
                    raise RuntimeError(f"Frame extraction failed: {error_msg}")

                await asyncio.sleep(poll_interval)

            # Get results
            results_response = await client.get(
                f"{self.frame_server_url}/frames/jobs/{job_id}/results"
            )
            results_response.raise_for_status()
            results = results_response.json()

            frames = results.get("frames", [])
            logger.info(f"Retrieved {len(frames)} frame metadata")
            return results

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return None

    def _calculate_scene_timestamps(
        self,
        scene_boundaries: List[Dict],
        frames_per_scene: int
    ) -> List[float]:
        """
        Calculate specific timestamps for scene-based extraction

        Extracts frames at beginning, middle, and end of each scene
        """
        timestamps = []

        for scene in scene_boundaries:
            start = scene.get("start_timestamp", 0.0)
            end = scene.get("end_timestamp", start + 1.0)
            duration = end - start

            if frames_per_scene == 1:
                # Single frame at scene midpoint
                timestamps.append(start + duration / 2)
            elif frames_per_scene == 2:
                # Start and end
                timestamps.append(start + duration * 0.1)
                timestamps.append(start + duration * 0.9)
            else:
                # Distribute evenly
                for i in range(frames_per_scene):
                    t = start + (duration * (i + 0.5) / frames_per_scene)
                    timestamps.append(t)

        return sorted(set(timestamps))

    async def get_frame(
        self,
        video_path: str,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Get a single frame at specified timestamp

        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds

        Returns:
            Frame as numpy array (H, W, C) in BGR format (OpenCV convention), or None
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.frame_server_url}/frames/extract-frame",
                params={
                    "video_path": video_path,
                    "timestamp": timestamp,
                    "output_format": "jpeg"
                }
            )
            response.raise_for_status()

            # Decode image from bytes (returns BGR)
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            return frame

        except Exception as e:
            logger.error(f"Error getting frame at {timestamp}s: {e}")
            return None

    async def get_frames_batch(
        self,
        video_path: str,
        timestamps: List[float],
        max_concurrent: int = 4
    ) -> List[Optional[np.ndarray]]:
        """
        Get multiple frames concurrently

        Args:
            video_path: Path to video file
            timestamps: List of timestamps to fetch
            max_concurrent: Maximum concurrent requests

        Returns:
            List of frames (None for failed fetches)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(ts: float) -> Optional[np.ndarray]:
            async with semaphore:
                return await self.get_frame(video_path, ts)

        tasks = [fetch_with_semaphore(ts) for ts in timestamps]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """Check if frame server is reachable"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.frame_server_url}/frames/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Frame server health check failed: {e}")
            return False
