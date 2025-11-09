"""
Frame Server - Frame Extractor
Extract frames from video using OpenCV or FFmpeg
"""

import cv2
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video files"""

    def __init__(self, extraction_method: str = "opencv_cuda"):
        """
        Initialize frame extractor

        Args:
            extraction_method: opencv_cuda, opencv_cpu, or ffmpeg
        """
        self.extraction_method = extraction_method
        self.temp_dir = Path("/tmp/frames")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, video_path: str) -> Tuple[float, float, int]:
        """
        Get video metadata

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (duration_seconds, fps, total_frames)
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            cap.release()

            logger.info(f"Video info: {duration:.2f}s, {fps:.2f} FPS, {total_frames} frames")
            return duration, fps, total_frames

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise

    def extract_frames_opencv(
        self,
        video_path: str,
        job_id: str,
        timestamps: List[float],
        output_format: str = "jpeg",
        quality: int = 95
    ) -> List[Tuple[int, float, str, int, int]]:
        """
        Extract frames using OpenCV

        Args:
            video_path: Path to video file
            job_id: Job identifier for temp file naming
            timestamps: List of timestamps to extract
            output_format: jpeg or png
            quality: Image quality (1-100)

        Returns:
            List of (index, timestamp, file_path, width, height)
        """
        frames = []

        try:
            # Enable CUDA if available and requested
            if self.extraction_method == "opencv_cuda":
                try:
                    cv2.ocl.setUseOpenCL(True)
                    logger.info("OpenCV CUDA acceleration enabled")
                except:
                    logger.warning("CUDA not available, falling back to CPU")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            job_dir = self.temp_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            for idx, timestamp in enumerate(timestamps):
                # Seek to timestamp
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Failed to read frame at {timestamp}s")
                    continue

                # Save frame
                ext = "jpg" if output_format == "jpeg" else "png"
                frame_path = job_dir / f"frame_{idx:06d}.{ext}"

                if output_format == "jpeg":
                    cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )
                else:
                    cv2.imwrite(str(frame_path), frame)

                frames.append((idx, timestamp, str(frame_path), width, height))

                if (idx + 1) % 100 == 0:
                    logger.info(f"Extracted {idx + 1}/{len(timestamps)} frames")

            cap.release()

            logger.info(f"Extraction complete: {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"Error extracting frames with OpenCV: {e}")
            raise

    def extract_frames_ffmpeg(
        self,
        video_path: str,
        job_id: str,
        timestamps: List[float],
        output_format: str = "jpeg",
        quality: int = 95
    ) -> List[Tuple[int, float, str, int, int]]:
        """
        Extract frames using FFmpeg (fallback method)

        Args:
            video_path: Path to video file
            job_id: Job identifier for temp file naming
            timestamps: List of timestamps to extract
            output_format: jpeg or png
            quality: Image quality (1-100)

        Returns:
            List of (index, timestamp, file_path, width, height)
        """
        frames = []

        try:
            job_dir = self.temp_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # Get video dimensions
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            width, height = map(int, result.stdout.strip().split(','))

            for idx, timestamp in enumerate(timestamps):
                ext = "jpg" if output_format == "jpeg" else "png"
                frame_path = job_dir / f"frame_{idx:06d}.{ext}"

                # Extract single frame at timestamp
                cmd = [
                    "ffmpeg",
                    "-ss", str(timestamp),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", str(max(1, 31 - int(quality / 3.23))),  # FFmpeg quality scale
                    "-y",
                    str(frame_path)
                ]

                subprocess.run(cmd, capture_output=True, check=True)

                if frame_path.exists():
                    frames.append((idx, timestamp, str(frame_path), width, height))

                if (idx + 1) % 100 == 0:
                    logger.info(f"Extracted {idx + 1}/{len(timestamps)} frames")

            logger.info(f"FFmpeg extraction complete: {len(frames)} frames")
            return frames

        except Exception as e:
            logger.error(f"Error extracting frames with FFmpeg: {e}")
            raise

    def generate_timestamps(
        self,
        duration: float,
        interval: float,
        scene_boundaries: Optional[List[dict]] = None
    ) -> List[float]:
        """
        Generate frame timestamps based on sampling strategy

        Args:
            duration: Video duration in seconds
            interval: Sampling interval in seconds
            scene_boundaries: Optional scene boundaries for scene-based sampling

        Returns:
            List of timestamps
        """
        timestamps = []

        if scene_boundaries:
            # Scene-based sampling: extract first, middle, last frame of each scene
            for scene in scene_boundaries:
                start = scene['start_timestamp']
                end = scene['end_timestamp']
                mid = (start + end) / 2

                timestamps.extend([start, mid, end])
        else:
            # Interval-based sampling
            current = 0.0
            while current < duration:
                timestamps.append(current)
                current += interval

        # Ensure timestamps don't exceed duration
        timestamps = [min(t, duration - 0.1) for t in timestamps if t < duration]

        logger.info(f"Generated {len(timestamps)} timestamps (interval: {interval}s)")
        return timestamps

    def extract(
        self,
        video_path: str,
        job_id: str,
        sampling_interval: float = 2.0,
        scene_boundaries: Optional[List[dict]] = None,
        output_format: str = "jpeg",
        quality: int = 95
    ) -> List[Tuple[int, float, str, int, int]]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            job_id: Job identifier
            sampling_interval: Interval between frames in seconds
            scene_boundaries: Optional scene boundaries for adaptive sampling
            output_format: jpeg or png
            quality: Image quality (1-100)

        Returns:
            List of (index, timestamp, file_path, width, height)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get video info
        duration, fps, total_frames = self.get_video_info(video_path)

        # Generate timestamps
        timestamps = self.generate_timestamps(duration, sampling_interval, scene_boundaries)

        # Extract frames
        if self.extraction_method.startswith("opencv"):
            return self.extract_frames_opencv(
                video_path, job_id, timestamps, output_format, quality
            )
        else:
            return self.extract_frames_ffmpeg(
                video_path, job_id, timestamps, output_format, quality
            )

    def cleanup_job(self, job_id: str):
        """
        Clean up temporary files for a job

        Args:
            job_id: Job identifier
        """
        try:
            job_dir = self.temp_dir / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up temp files for job: {job_id}")
        except Exception as e:
            logger.error(f"Error cleaning up job {job_id}: {e}")
