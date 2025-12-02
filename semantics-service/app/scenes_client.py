"""
Semantics Service - Scenes Service Client
Client for fetching pre-computed scene boundaries from scenes-service
"""

import os
import httpx
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

SCENES_SERVER_URL = os.getenv("SCENES_SERVER_URL", "http://scenes-service:5002")


class ScenesServerClient:
    """Client for scenes-service integration"""

    def __init__(self, base_url: str = SCENES_SERVER_URL):
        """
        Initialize scenes service client

        Args:
            base_url: Base URL for scenes-service (default from env)
        """
        self.base_url = base_url
        logger.info(f"ScenesServerClient initialized with base_url: {base_url}")

    async def get_scene_boundaries(self, job_id: str) -> List[Dict[str, float]]:
        """
        Fetch scene boundaries from completed scenes job

        Args:
            job_id: Job ID from scenes-service

        Returns:
            List of scene boundary dicts with start_timestamp and end_timestamp

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If job is not completed or has no results
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First check job status
                status_url = f"{self.base_url}/scenes/jobs/{job_id}/status"
                logger.info(f"Fetching scenes job status: {status_url}")

                status_response = await client.get(status_url)
                status_response.raise_for_status()
                status_data = status_response.json()

                if status_data.get("status") != "completed":
                    raise ValueError(
                        f"Scenes job {job_id} is not completed (status: {status_data.get('status')})"
                    )

                # Fetch results
                results_url = f"{self.base_url}/scenes/jobs/{job_id}/results"
                logger.info(f"Fetching scenes job results: {results_url}")

                results_response = await client.get(results_url)
                results_response.raise_for_status()
                results_data = results_response.json()

                # Extract scenes from results
                scenes = results_data.get("scenes", [])
                if not scenes:
                    logger.warning(f"No scenes found in job {job_id}")
                    return []

                # Convert to boundary format expected by semantics service
                boundaries = [
                    {
                        "start_timestamp": scene.get("start_time", 0.0),
                        "end_timestamp": scene.get("end_time", 0.0)
                    }
                    for scene in scenes
                ]

                logger.info(f"Retrieved {len(boundaries)} scene boundaries from job {job_id}")
                return boundaries

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching scenes for job {job_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching scenes for job {job_id}: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if scenes-service is reachable

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/scenes/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Scenes service health check failed: {e}")
            return False
