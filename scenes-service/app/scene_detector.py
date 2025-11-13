"""
Scenes Service - Scene Detector
PySceneDetect with CUDA-accelerated OpenCV
"""

import cv2
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode

logger = logging.getLogger(__name__)


class SceneDetector:
    """Detect scene boundaries in video using PySceneDetect"""

    def __init__(self, opencv_device: str = "cuda"):
        """
        Initialize scene detector

        Args:
            opencv_device: cuda or cpu
        """
        self.opencv_device = opencv_device

        # Enable CUDA if available
        if opencv_device == "cuda":
            try:
                cv2.ocl.setUseOpenCL(True)
                logger.info("OpenCV CUDA acceleration enabled")
            except Exception as e:
                logger.warning(f"Failed to enable CUDA: {e}, falling back to CPU")

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

    def detect_scenes(
        self,
        video_path: str,
        detection_method: str = "content",
        scene_threshold: float = 27.0,
        min_scene_length: float = 0.6
    ) -> List[Tuple[int, int, float, float]]:
        """
        Detect scene boundaries in video

        Args:
            video_path: Path to video file
            detection_method: content, threshold, or adaptive
            scene_threshold: Detection threshold (method-specific)
            min_scene_length: Minimum scene length in seconds

        Returns:
            List of (start_frame, end_frame, start_time, end_time)
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")

            logger.info(f"Starting scene detection: {video_path}")
            logger.info(f"Method: {detection_method}, Threshold: {scene_threshold}, Min length: {min_scene_length}s")

            start_time = time.time()

            # Open video
            video = open_video(video_path)

            # Create scene manager
            scene_manager = SceneManager()

            # Add detector based on method
            if detection_method == "content":
                detector = ContentDetector(threshold=scene_threshold, min_scene_len=min_scene_length)
            elif detection_method == "threshold":
                detector = ThresholdDetector(threshold=scene_threshold, min_scene_len=min_scene_length)
            elif detection_method == "adaptive":
                detector = AdaptiveDetector(adaptive_threshold=scene_threshold, min_scene_len=min_scene_length)
            else:
                raise ValueError(f"Unknown detection method: {detection_method}")

            scene_manager.add_detector(detector)

            # Detect scenes
            scene_manager.detect_scenes(video)

            # Get scene list
            scene_list = scene_manager.get_scene_list()

            # Convert to result format
            scenes = []
            for scene_num, (start_timecode, end_timecode) in enumerate(scene_list):
                start_frame = start_timecode.get_frames()
                end_frame = end_timecode.get_frames()
                start_time_sec = start_timecode.get_seconds()
                end_time_sec = end_timecode.get_seconds()

                scenes.append((start_frame, end_frame, start_time_sec, end_time_sec))

            processing_time = time.time() - start_time

            logger.info(f"Scene detection complete: {len(scenes)} scenes in {processing_time:.2f}s")

            # Log scene statistics
            if scenes:
                durations = [end_time - start_time for _, _, start_time, end_time in scenes]
                avg_duration = sum(durations) / len(durations)
                logger.info(f"Average scene duration: {avg_duration:.2f}s")

            return scenes

        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            raise

    def estimate_scene_count(
        self,
        duration: float,
        avg_scene_length: float = 3.0
    ) -> int:
        """
        Estimate number of scenes based on video duration

        Args:
            duration: Video duration in seconds
            avg_scene_length: Assumed average scene length

        Returns:
            Estimated scene count
        """
        if duration <= 0:
            return 0

        # Conservative estimate
        estimated = int(duration / avg_scene_length)
        return max(1, estimated)

    def scenes_to_boundaries(
        self,
        scenes: List[Tuple[int, int, float, float]]
    ) -> List[dict]:
        """
        Convert scene tuples to boundary dictionaries

        Args:
            scenes: List of (start_frame, end_frame, start_time, end_time)

        Returns:
            List of scene boundary dicts
        """
        boundaries = []

        for scene_num, (start_frame, end_frame, start_time, end_time) in enumerate(scenes):
            boundary = {
                "scene_number": scene_num,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "duration": end_time - start_time
            }
            boundaries.append(boundary)

        return boundaries

    def get_detector_info(self) -> dict:
        """
        Get detector configuration information

        Returns:
            Dict with detector info
        """
        return {
            "opencv_device": self.opencv_device,
            "cuda_enabled": cv2.ocl.useOpenCL() if self.opencv_device == "cuda" else False,
            "available_detectors": ["content", "threshold", "adaptive"]
        }
