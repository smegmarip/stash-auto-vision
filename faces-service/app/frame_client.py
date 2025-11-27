"""
Faces Service - Frame Server Client
HTTP client for calling frame-server enhancement endpoints
"""

import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FrameServerClient:
    """Client for frame-server enhancement API"""

    def __init__(self, frame_server_url: str):
        """
        Initialize frame-server client

        Args:
            frame_server_url: Base URL of frame-server (e.g., http://frame-server:5001)
        """
        self.base_url = frame_server_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)  # 60s timeout for enhancement
        logger.info(f"Frame server client initialized: {self.base_url}")

    async def enhance_frame(
        self,
        video_path: str,
        timestamp: float,
        model: str = "codeformer",
        fidelity_weight: float = 0.5,
        output_format: str = "jpeg",
        quality: int = 95
    ) -> Optional[bytes]:
        """
        Enhance a single frame via frame-server

        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            model: Enhancement model ('gfpgan' or 'codeformer')
            fidelity_weight: Fidelity vs quality (0.0-1.0)
            output_format: Output format ('jpeg' or 'png')
            quality: JPEG quality (1-100)

        Returns:
            Enhanced frame as bytes, or None if failed
        """
        try:
            url = f"{self.base_url}/extract-frame"
            params = {
                "video_path": video_path,
                "timestamp": timestamp,
                "enhance": 1,
                "model": model,
                "fidelity_weight": fidelity_weight,
                "output_format": output_format,
                "quality": quality
            }

            logger.debug(f"Requesting enhanced frame: {video_path} @ {timestamp}s")
            response = await self.client.get(url, params=params, timeout=600.0)
            response.raise_for_status()

            logger.info(f"Enhanced frame received: {len(response.content)} bytes")
            return response.content

        except httpx.HTTPStatusError as e:
            logger.error(f"Frame server HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Frame server request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Frame server unexpected error: {e}", exc_info=True)
            return None

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
        logger.info("Frame server client closed")
