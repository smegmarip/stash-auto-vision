"""
Captioning Service - Frame Server Client
Client for fetching frames from frame-server service with sharpness-based selection
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
import httpx
import numpy as np
import cv2
import logging

from .sharpness import (
    calculate_laplacian_variance,
    calculate_combined_quality,
    select_sharpest_per_scene
)

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
        poll_interval: float = 1.0,
        select_sharpest: bool = False,
        candidate_multiplier: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Extract frames from video via frame-server

        Args:
            video_path: Path to video file
            sampling_interval: Interval between frames in seconds (interval mode)
            scene_boundaries: Optional scene boundaries for scene-based extraction
            frames_per_scene: Frames to extract per scene (final count after filtering)
            poll_interval: Seconds between status polls
            select_sharpest: If True, extract more candidates and select sharpest
            candidate_multiplier: Extract N*frames_per_scene candidates for sharpness selection

        Returns:
            Frame extraction result with metadata
        """
        client = await self._get_client()

        try:
            # Calculate how many candidates to extract
            candidates_per_scene = frames_per_scene
            if select_sharpest and scene_boundaries:
                candidates_per_scene = frames_per_scene * candidate_multiplier
                logger.info(f"Sharpest selection enabled: extracting {candidates_per_scene} candidates per scene")

            # If scene boundaries provided, extract specific timestamps
            if scene_boundaries:
                timestamps = self._calculate_scene_timestamps(
                    scene_boundaries, candidates_per_scene
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

            frames_metadata = results.get("frames", [])
            logger.info(f"Retrieved {len(frames_metadata)} frame metadata")

            # If sharpness selection enabled, filter to sharpest frames
            if select_sharpest and scene_boundaries and len(frames_metadata) > frames_per_scene * len(scene_boundaries):
                results = await self._select_sharpest_frames(
                    video_path,
                    results,
                    scene_boundaries,
                    frames_per_scene
                )

            return results

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return None

    async def _select_sharpest_frames(
        self,
        video_path: str,
        results: Dict[str, Any],
        scene_boundaries: List[Dict],
        frames_per_scene: int
    ) -> Dict[str, Any]:
        """
        Filter extraction results to keep only the sharpest frames per scene.

        Args:
            video_path: Path to video file
            results: Frame extraction results from frame-server
            scene_boundaries: Scene boundary information
            frames_per_scene: Number of frames to keep per scene

        Returns:
            Filtered results with only sharpest frames
        """
        frames_metadata = results.get("frames", [])
        if not frames_metadata:
            return results

        timestamps = [f["timestamp"] for f in frames_metadata]

        logger.info(f"Loading {len(timestamps)} candidate frames for sharpness analysis...")

        # Fetch all candidate frames
        frames = await self.get_frames_batch(video_path, timestamps, max_concurrent=8)

        # Select sharpest per scene
        selected = select_sharpest_per_scene(
            frames=frames,
            timestamps=timestamps,
            scene_boundaries=scene_boundaries,
            frames_per_scene=frames_per_scene,
            use_combined_score=True
        )

        if not selected:
            logger.warning("Sharpness selection returned no results, keeping original frames")
            return results

        # Filter metadata to selected frames
        selected_indices = {s[0] for s in selected}
        filtered_metadata = [
            frames_metadata[i] for i in sorted(selected_indices)
        ]

        # Add sharpness scores to metadata
        score_map = {s[0]: s[2] for s in selected}
        for meta in filtered_metadata:
            idx = frames_metadata.index(meta)
            if idx in score_map:
                meta["sharpness_score"] = score_map[idx]

        logger.info(f"Selected {len(filtered_metadata)} sharpest frames from {len(frames_metadata)} candidates")

        # Update results
        results["frames"] = filtered_metadata
        results["metadata"]["frames_analyzed"] = len(frames_metadata)
        results["metadata"]["sharpness_filtered"] = True

        return results

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

    async def extract_sprites(
        self,
        video_path: str,
        sprite_vtt_url: str,
        sprite_image_url: str,
        frames_per_scene: int = 3,
        scene_boundaries: Optional[List[Dict]] = None,
        select_sharpest: bool = True,
        poll_interval: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Extract frames from sprite sheet via frame-server.

        Ultra-fast frame extraction from pre-generated sprite grids.
        Bypasses video decoding entirely.

        Args:
            video_path: Path to video (for metadata reference)
            sprite_vtt_url: URL to WebVTT file with sprite coordinates
            sprite_image_url: URL to sprite grid JPEG image
            frames_per_scene: Frames to select per scene
            scene_boundaries: Optional scene boundaries for filtering
            select_sharpest: Whether to filter by sharpness
            poll_interval: Seconds between status polls

        Returns:
            Frame extraction result with metadata
        """
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.frame_server_url}/frames/extract",
                json={
                    "video_path": video_path,
                    "use_sprites": True,
                    "sprite_vtt_url": sprite_vtt_url,
                    "sprite_image_url": sprite_image_url,
                    "output_format": "jpeg",
                    "quality": 95
                }
            )
            response.raise_for_status()
            job_info = response.json()
            job_id = job_info.get("job_id")

            if not job_id:
                return job_info

            logger.info(f"Sprite extraction job submitted: {job_id}")

            # Poll for completion
            while True:
                status_response = await client.get(
                    f"{self.frame_server_url}/frames/jobs/{job_id}/status"
                )
                status = status_response.json()

                if status.get("status") == "completed":
                    logger.info(f"Sprite extraction completed: {job_id}")
                    break
                elif status.get("status") == "failed":
                    error_msg = status.get("error", "Unknown error")
                    raise RuntimeError(f"Sprite extraction failed: {error_msg}")

                await asyncio.sleep(poll_interval)

            # Get results
            results_response = await client.get(
                f"{self.frame_server_url}/frames/jobs/{job_id}/results"
            )
            results_response.raise_for_status()
            results = results_response.json()

            frames_metadata = results.get("frames", [])
            logger.info(f"Retrieved {len(frames_metadata)} sprite frames")

            # Filter by scene boundaries if provided
            if scene_boundaries:
                results = self._filter_by_scenes(
                    results, scene_boundaries, frames_per_scene
                )

            # Apply sharpness selection if enabled
            if select_sharpest and scene_boundaries:
                results = await self._select_sharpest_frames(
                    video_path,
                    results,
                    scene_boundaries,
                    frames_per_scene
                )

            return results

        except Exception as e:
            logger.error(f"Error extracting sprites: {e}")
            return None

    def _filter_by_scenes(
        self,
        results: Dict[str, Any],
        scene_boundaries: List[Dict],
        frames_per_scene: int
    ) -> Dict[str, Any]:
        """
        Filter sprite frames to keep only those within scene boundaries.

        Args:
            results: Frame extraction results
            scene_boundaries: Scene boundary information
            frames_per_scene: Max frames to keep per scene

        Returns:
            Filtered results
        """
        frames_metadata = results.get("frames", [])
        if not frames_metadata:
            return results

        filtered = []

        for scene in scene_boundaries:
            start = scene.get("start_timestamp", 0.0)
            end = scene.get("end_timestamp", float('inf'))

            # Find frames in this scene
            scene_frames = [
                f for f in frames_metadata
                if start <= f["timestamp"] <= end
            ]

            # Distribute evenly if we have more than needed
            if len(scene_frames) > frames_per_scene:
                step = len(scene_frames) / frames_per_scene
                indices = [int(i * step) for i in range(frames_per_scene)]
                scene_frames = [scene_frames[i] for i in indices]

            filtered.extend(scene_frames)

        results["frames"] = filtered
        results["metadata"]["scene_filtered"] = True

        logger.info(f"Filtered to {len(filtered)} frames from scenes")
        return results

    async def health_check(self) -> bool:
        """Check if frame server is reachable"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.frame_server_url}/frames/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Frame server health check failed: {e}")
            return False
