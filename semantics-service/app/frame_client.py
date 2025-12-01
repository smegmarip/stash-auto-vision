"""
Semantics Service - Frame Server Client
HTTP client for extracting frames from videos
"""

import httpx
import asyncio
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from app.models import FramesExtractionResult

logger = logging.getLogger(__name__)


class FrameServerClient:
    """Client for frame-server extraction API"""

    def __init__(self, frame_server_url: str):
        """
        Initialize frame-server client

        Args:
            frame_server_url: Base URL of frame-server (e.g., http://frame-server:5001)
        """
        self.base_url = frame_server_url.rstrip("/")
        logger.info(f"Frame server client initialized: {self.base_url}")

    async def extract_frames(
        self,
        video_path: str,
        sampling_interval: float = 2.0,
        method: str = "opencv",
        timeout: int = 300
    ) -> FramesExtractionResult:
        """
        Extract frames from video via frame-server

        Args:
            video_path: Path to video file
            sampling_interval: Sampling interval in seconds
            method: Extraction method
            timeout: Request timeout in seconds

        Returns:
            List of frame metadata dicts with timestamps and frame indices
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Submit frame extraction job
                url = f"{self.base_url}/extract"
                payload = {
                    "video_path": video_path,
                    "sampling_interval": sampling_interval,
                    "method": method
                }

                logger.info(f"Requesting frame extraction: {video_path}")
                response = await client.post(url, json=payload)
                response.raise_for_status()

                job_info = response.json()
                job_id = job_info["job_id"]
                logger.info(f"Frame extraction job submitted: {job_id}")

                # Poll for completion
                while True:
                    status_url = f"{self.base_url}/jobs/{job_id}/status"
                    status_response = await client.get(status_url)
                    status = status_response.json()

                    if status["status"] == "completed":
                        logger.info(f"Frame extraction completed: {job_id}")
                        break
                    elif status["status"] == "failed":
                        raise RuntimeError(f"Frame extraction failed: {status.get('error')}")

                    await asyncio.sleep(1)

                # Get results
                results_url = f"{self.base_url}/jobs/{job_id}/results"
                results_response = await client.get(results_url)
                results = results_response.json()

                frames = results.get("frames", [])
                logger.info(f"Retrieved {len(frames)} frame metadata")
                return FramesExtractionResult(**results)

        except httpx.HTTPStatusError as e:
            logger.error(f"Frame server HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Frame server request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Frame server unexpected error: {e}", exc_info=True)
            raise

    async def get_frame_image(
        self,
        video_path: str,
        timestamp: float,
        timeout: int = 30
    ) -> Optional[np.ndarray]:
        """
        Get a single frame as numpy array

        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            timeout: Request timeout in seconds

        Returns:
            Frame as numpy array (H, W, C) or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                url = f"{self.base_url}/extract-frame"
                params = {
                    "video_path": video_path,
                    "timestamp": timestamp,
                    "output_format": "jpeg"
                }

                logger.debug(f"Requesting frame: {video_path} @ {timestamp}s")
                response = await client.get(url, params=params)
                response.raise_for_status()

                # Decode image bytes to numpy array
                image_bytes = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                return frame

        except Exception as e:
            logger.error(f"Error getting frame image: {e}")
            return None

    async def get_frames_batch(
        self,
        video_path: str,
        timestamps: List[float],
        timeout: int = 300
    ) -> List[Optional[np.ndarray]]:
        """
        Get multiple frames as numpy arrays

        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            timeout: Request timeout in seconds

        Returns:
            List of frames as numpy arrays (some may be None if failed)
        """
        frames = []

        for timestamp in timestamps:
            frame = await self.get_frame_image(video_path, timestamp, timeout)
            frames.append(frame)

        logger.info(f"Retrieved {sum(1 for f in frames if f is not None)}/{len(frames)} frames")
        return frames
