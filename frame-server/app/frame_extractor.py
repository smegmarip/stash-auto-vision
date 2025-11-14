"""
Frame Server - Frame Extractor
Extract frames from video using OpenCV, PyAV, or FFmpeg
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

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    av = None

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video files"""

    def __init__(self, extraction_method: str = "opencv_cuda", enable_fallback: bool = True):
        """
        Initialize frame extractor

        Args:
            extraction_method: opencv_cuda, opencv_cpu, pyav_hw, pyav_sw, or ffmpeg
            enable_fallback: Enable automatic fallback to other methods on failure
        """
        self.extraction_method = extraction_method
        self.enable_fallback = enable_fallback
        self.temp_dir = Path("/tmp/frames")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Define fallback chain
        self.fallback_methods = self._build_fallback_chain(extraction_method)

        if PYAV_AVAILABLE:
            logger.info("PyAV available for robust frame extraction")
        else:
            logger.warning("PyAV not available - limited to OpenCV and FFmpeg")

    def _build_fallback_chain(self, primary_method: str) -> List[str]:
        """Build fallback method chain based on primary method"""
        chain = [primary_method]

        if not self.enable_fallback:
            return chain

        # Add PyAV methods if available
        if PYAV_AVAILABLE:
            if primary_method != "pyav_hw":
                chain.append("pyav_hw")
            if primary_method != "pyav_sw":
                chain.append("pyav_sw")

        # Add FFmpeg CLI as last resort
        if primary_method != "ffmpeg":
            chain.append("ffmpeg")

        logger.info(f"Fallback chain: {' -> '.join(chain)}")
        return chain

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

    def extract_frame_pyav(
        self,
        video_path: str,
        timestamp: float,
        hw_accel: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame using PyAV

        Args:
            video_path: Path to video file
            timestamp: Timestamp to extract
            hw_accel: Enable hardware acceleration

        Returns:
            Frame as numpy array (BGR format) or None on failure
        """
        if not PYAV_AVAILABLE:
            return None

        try:
            options = {}
            if hw_accel:
                # Try hardware acceleration (CUDA, VAAPI, etc.)
                options['hwaccel'] = 'auto'
                options['hwaccel_output_format'] = 'auto'

            container = av.open(video_path, options=options)
            video_stream = container.streams.video[0]

            # Calculate target PTS (presentation timestamp)
            target_pts = int(timestamp / float(video_stream.time_base))

            # Seek to just before the target timestamp
            # Use backward seek to ensure we don't miss the frame
            seek_pts = max(0, target_pts - int(5 / float(video_stream.time_base)))
            container.seek(seek_pts, stream=video_stream)

            # Decode frames until we find the closest match
            closest_frame = None
            closest_diff = float('inf')

            for frame in container.decode(video=0):
                frame_time = float(frame.pts * video_stream.time_base)
                time_diff = abs(frame_time - timestamp)

                if time_diff < closest_diff:
                    closest_diff = time_diff
                    closest_frame = frame

                # If we've gone past the target, stop
                if frame_time > timestamp + 0.5:
                    break

                # If we found an exact match (within 1 frame), stop
                if time_diff < (1.0 / video_stream.average_rate):
                    break

            container.close()

            if closest_frame is not None:
                # Convert to numpy array in BGR format (OpenCV compatible)
                img = closest_frame.to_ndarray(format='bgr24')
                return img

            return None

        except Exception as e:
            logger.debug(f"PyAV extraction failed at {timestamp}s: {e}")
            return None

    def extract_frame_opencv_single(
        self,
        video_path: str,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Extract a single frame using OpenCV

        Args:
            video_path: Path to video file
            timestamp: Timestamp to extract

        Returns:
            Frame as numpy array (BGR format) or None on failure
        """
        try:
            # Enable CUDA if available and requested
            if self.extraction_method == "opencv_cuda":
                try:
                    cv2.ocl.setUseOpenCL(True)
                except:
                    pass

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return None

            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            ret, frame = cap.read()
            cap.release()

            if ret:
                return frame

            return None

        except Exception as e:
            logger.debug(f"OpenCV extraction failed at {timestamp}s: {e}")
            return None

    def extract_frame_ffmpeg_single(
        self,
        video_path: str,
        timestamp: float,
        output_path: str
    ) -> bool:
        """
        Extract a single frame using FFmpeg CLI

        Args:
            video_path: Path to video file
            timestamp: Timestamp to extract
            output_path: Where to save the frame

        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10
            )

            return result.returncode == 0 and os.path.exists(output_path)

        except Exception as e:
            logger.debug(f"FFmpeg extraction failed at {timestamp}s: {e}")
            return False

    def extract_single_frame_with_fallback(
        self,
        video_path: str,
        timestamp: float,
        job_id: str,
        idx: int,
        output_format: str = "jpeg",
        quality: int = 95
    ) -> Optional[Tuple[int, float, str, int, int]]:
        """
        Extract a single frame with automatic fallback across methods

        Args:
            video_path: Path to video file
            timestamp: Timestamp to extract
            job_id: Job identifier
            idx: Frame index
            output_format: jpeg or png
            quality: Image quality (1-100)

        Returns:
            Tuple of (index, timestamp, file_path, width, height) or None
        """
        job_dir = self.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        ext = "jpg" if output_format == "jpeg" else "png"
        frame_path = job_dir / f"frame_{idx:06d}.{ext}"

        for method in self.fallback_methods:
            frame = None

            try:
                if method == "opencv_cuda" or method == "opencv_cpu":
                    frame = self.extract_frame_opencv_single(video_path, timestamp)

                elif method == "pyav_hw":
                    frame = self.extract_frame_pyav(video_path, timestamp, hw_accel=True)

                elif method == "pyav_sw":
                    frame = self.extract_frame_pyav(video_path, timestamp, hw_accel=False)

                elif method == "ffmpeg":
                    if self.extract_frame_ffmpeg_single(video_path, timestamp, str(frame_path)):
                        # FFmpeg already saved the file, just get dimensions
                        img = cv2.imread(str(frame_path))
                        if img is not None:
                            height, width = img.shape[:2]
                            return (idx, timestamp, str(frame_path), width, height)

                # If we got a frame from OpenCV/PyAV, save it
                if frame is not None:
                    if output_format == "jpeg":
                        cv2.imwrite(
                            str(frame_path),
                            frame,
                            [cv2.IMWRITE_JPEG_QUALITY, quality]
                        )
                    else:
                        cv2.imwrite(str(frame_path), frame)

                    height, width = frame.shape[:2]
                    logger.debug(f"Frame {idx} extracted successfully with {method}")
                    return (idx, timestamp, str(frame_path), width, height)

            except Exception as e:
                logger.debug(f"Method {method} failed for frame {idx}: {e}")
                continue

        # All methods failed
        logger.warning(f"All extraction methods failed for frame at {timestamp}s")
        return None

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
        Extract frames from video with automatic fallback

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

        # Extract frames with per-frame fallback if enabled
        if self.enable_fallback:
            logger.info(f"Extracting {len(timestamps)} frames with fallback enabled")
            frames = []

            for idx, timestamp in enumerate(timestamps):
                frame_result = self.extract_single_frame_with_fallback(
                    video_path, timestamp, job_id, idx, output_format, quality
                )

                if frame_result is not None:
                    frames.append(frame_result)

                if (idx + 1) % 100 == 0:
                    logger.info(f"Extracted {len(frames)}/{idx + 1} frames ({len(frames) / (idx + 1) * 100:.1f}% success)")

            logger.info(f"Extraction complete: {len(frames)}/{len(timestamps)} frames ({len(frames) / len(timestamps) * 100:.1f}% success)")
            return frames

        # Legacy single-method extraction
        else:
            logger.info(f"Extracting {len(timestamps)} frames with {self.extraction_method} (no fallback)")
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
